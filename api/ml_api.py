"""
Mandi Intelligence API (v4).

Run from the mandi_intelligence/ directory:
    uvicorn api.ml_api:app --port 8000

Startup loads the saved models (training only if they are missing or were
trained on different data) and pre-computes forecasts, so /recommend is just
profit arithmetic.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_arbitrage.arbitrage_engine import get_petrol_price, set_petrol_price
from ml_arbitrage.data_loader import TARGET_CROPS
from ml_arbitrage.distance_calculator import get_distance, is_known_location, known_locations
from ml_arbitrage.service import MandiService

logger = logging.getLogger("mandi_api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_state: Dict = {"service": None, "startup_err": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["service"] = await asyncio.to_thread(MandiService.start)
    except Exception as exc:
        _state["startup_err"] = str(exc)
        logger.exception("Startup failed: %s", exc)
    yield


app = FastAPI(
    title="Mandi Intelligence API",
    description="ML price forecasts and net-profit-optimised mandi recommendations.",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ── Request / response models ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    crop: str = Field(..., description="Crop name (Onion/Tomato/Potato)")
    quantity: float = Field(..., gt=0, le=100_000, description="Quantity in kg")
    farmer_location: str = Field("Gandhinagar", description="Farmer's town in Gujarat")
    petrol_price: Optional[float] = Field(None, gt=0, le=500,
                                          description="Petrol price in Rs/L (default if omitted)")

    @field_validator("crop")
    @classmethod
    def crop_must_be_valid(cls, v: str) -> str:
        for c in TARGET_CROPS:
            if c.lower() == v.strip().lower():
                return c
        raise ValueError(f"Invalid crop '{v}'. Choose from: {TARGET_CROPS}")

    @field_validator("farmer_location")
    @classmethod
    def location_must_be_known(cls, v: str) -> str:
        v = v.strip()[:64]
        if not is_known_location(v):
            raise ValueError(f"Unknown location '{v}'. Supported: {', '.join(known_locations())}")
        return v


class TransportDetail(BaseModel):
    vehicle: str
    trips: int = 1
    petrol_rate: float = 94.0
    fuel: float
    driver: float
    toll: float
    maintenance: float
    loading_unloading: float


class CostBreakdown(BaseModel):
    gross_revenue: float
    transport_cost: float
    transport_detail: TransportDetail
    storage_cost: float
    perishability_cost: float
    qty_remaining_kg: float


class OptimalStrategy(BaseModel):
    mandi: str
    mandi_name: str              # frontend alias
    distance_km: float
    price_per_kg: float
    current_price: float         # frontend alias
    days_to_wait: int
    is_predicted_price: bool
    action_label: str
    recommendation: str          # frontend alias (descriptive)
    net_profit: float
    net_profit_lower: float
    net_profit_upper: float
    margin_pct: float
    gross_revenue: float
    transport_cost: float
    storage_cost: float
    predicted_price_7d: Optional[float] = None
    cost_breakdown: CostBreakdown


class RecommendResponse(BaseModel):
    request_id: str
    crop: str
    quantity_kg: float
    quantity: float              # frontend alias
    farmer_location: str
    recommendation: str
    summary: str                 # frontend alias
    justification: str
    optimal_strategy: OptimalStrategy
    best_option: OptimalStrategy  # frontend alias
    alternative_count: int
    alternatives: List[Dict]
    model_report_url: str


class FeedbackRequest(BaseModel):
    farmer_id: Optional[str] = None
    mandi_name: str
    crop: str
    quantity: float = Field(..., gt=0)
    actual_price: float = Field(..., gt=0)
    sale_date: str
    feedback: Optional[str] = None


# ── Middleware & helpers ─────────────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Response-Time"] = f"{ms:.1f}ms"
    logger.info("[%s] %s %s → %d (%.1f ms)", request.state.request_id,
                request.method, request.url.path, response.status_code, ms)
    return response


def _service() -> MandiService:
    svc = _state["service"]
    if svc is None:
        raise HTTPException(status_code=503,
                            detail=_state["startup_err"] or "System not ready — startup in progress.")
    return svc


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"service": "Mandi Intelligence API v4",
            "status": "ready" if _state["service"] else "loading", "docs": "/docs"}


@app.get("/health")
async def health():
    svc = _state["service"]
    return {
        "status": "healthy" if svc else "degraded",
        "startup_error": _state["startup_err"],
        "models_loaded": svc is not None,
        "records": len(svc.panel) if svc else 0,
        "mandis": int(svc.active["Mandi_Name"].nunique()) if svc else 0,
        "crops": svc.crops if svc else [],
        "data_end": str(svc.panel["Date"].max().date()) if svc else None,
    }


@app.get("/mandis")
async def list_mandis():
    svc = _service()
    result = []
    for mandi, g in svc.active.groupby("Mandi_Name"):
        result.append({
            "mandi_name": mandi,
            "distance_km": get_distance("Gandhinagar", mandi),
            "available_crops": sorted(g["Crop"].tolist()),
            "record_count": int(g["n_obs"].sum()),
            "last_report": str(g["last_date"].max().date()),
        })
    return {"total_mandis": len(result), "mandis": sorted(result, key=lambda x: x["distance_km"])}


@app.get("/locations")
async def list_locations():
    return {"locations": known_locations()}


@app.get("/model_report")
async def model_report():
    return _service().forecaster.report()


@app.get("/config/petrol")
async def get_petrol():
    return {"petrol_price_per_litre": get_petrol_price()}


@app.post("/config/petrol")
async def update_petrol(body: dict):
    """Update the default petrol price (affects all later recommendations)."""
    price = body.get("price_per_litre")
    if not isinstance(price, (int, float)) or not 0 < price <= 500:
        raise HTTPException(status_code=422, detail="price_per_litre must be between 0 and 500")
    set_petrol_price(float(price))
    return {"status": "updated", "petrol_price_per_litre": get_petrol_price(),
            "message": f"Petrol price set to Rs {price:.2f}/L for all future calculations."}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest, request: Request):
    """Best mandi and day to sell, by net profit after transport, storage and spoilage."""
    svc = _service()
    try:
        result = svc.recommend(body.crop, body.quantity, body.farmer_location, body.petrol_price)
    except Exception as exc:
        logger.exception("Strategy generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    opt = result["optimal_strategy"]
    bd = opt["cost_breakdown"]
    fc = svc.forecast(body.crop)
    day7 = fc[(fc["Mandi_Name"] == opt["mandi"]) & (fc["Day_Ahead"] == 7)]
    p_t7 = round(float(day7["Predicted_Price"].iloc[0]), 2) if not day7.empty else None

    strategy = OptimalStrategy(
        mandi=opt["mandi"],
        mandi_name=opt["mandi"],
        distance_km=opt["distance_km"],
        price_per_kg=opt["price_per_kg"],
        current_price=opt["price_per_kg"],
        days_to_wait=opt["days_to_wait"],
        is_predicted_price=opt["is_predicted_price"],
        action_label=opt["action_label"],
        recommendation=result["recommendation"],
        net_profit=opt["net_profit"],
        net_profit_lower=opt["net_profit_lower"],
        net_profit_upper=opt["net_profit_upper"],
        margin_pct=opt["margin_pct"],
        gross_revenue=bd["gross_revenue"],
        transport_cost=bd["transport_cost"],
        storage_cost=bd["storage_cost"],
        predicted_price_7d=p_t7,
        cost_breakdown=CostBreakdown(
            gross_revenue=bd["gross_revenue"],
            transport_cost=bd["transport_cost"],
            transport_detail=TransportDetail(**bd["transport_detail"]),
            storage_cost=bd["storage_cost"],
            perishability_cost=bd["perishability_cost"],
            qty_remaining_kg=bd["qty_remaining_kg"],
        ),
    )
    alternatives = [{
        "mandi_name": s["mandi_name"],
        "distance_km": s["distance_km"],
        "current_price": s["price_per_kg"],
        "net_profit": s["net_profit"],
        "recommendation": s["action_label"],
        "days_to_wait": s["days_to_wait"],
    } for s in result["alternative_scenarios"]]

    return RecommendResponse(
        request_id=request.state.request_id,
        crop=body.crop,
        quantity_kg=body.quantity,
        quantity=body.quantity,
        farmer_location=body.farmer_location,
        recommendation=result["recommendation"],
        summary=result["recommendation"],
        justification=result["justification"],
        optimal_strategy=strategy,
        best_option=strategy,
        alternative_count=len(alternatives),
        alternatives=alternatives,
        model_report_url="/model_report",
    )


@app.post("/response", response_model=RecommendResponse)
async def response_alias(body: RecommendRequest, request: Request):
    """Alias for /recommend (v2 clients)."""
    return await recommend(body, request)


@app.post("/respond")
async def submit_feedback(body: FeedbackRequest):
    """Record an actual sale (logged for now; intended for future retraining)."""
    farmer_id = body.farmer_id or "FARMER_" + hashlib.sha1(
        f"{body.mandi_name}|{body.sale_date}".encode()).hexdigest()[:6].upper()
    logger.info("Feedback [%s]: %s %s kg @ ₹%.2f on %s at %s", farmer_id, body.crop,
                body.quantity, body.actual_price, body.sale_date, body.mandi_name)
    return {"status": "success", "farmer_id": farmer_id,
            "message": "Sale data recorded. Thank you — this improves recommendations for all farmers."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.ml_api:app", host="0.0.0.0", port=8000)
