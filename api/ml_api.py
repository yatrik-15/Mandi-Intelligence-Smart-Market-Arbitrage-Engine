"""
api/ml_api.py — Mandi Intelligence FastAPI v3.0
================================================
FIXES FROM v2.x
---------------
  ❌ BUG: `@app.on_event("startup")` is deprecated since FastAPI 0.93;
          replaced with `lifespan` context manager.
  ❌ BUG: On startup failure, global `engine = None` caused every endpoint
          to return 503 with no diagnostic — now returns a degraded-mode
          response with the actual error.
  ❌ BUG: `/response` and `/recommend` were different endpoints doing the
          same thing (duplicate code). Unified under `/recommend`; `/response`
          aliased for backward compatibility.
  ❌ BUG: `opt.get("is_predicted_price")` key mismatch with engine output
          dict key `is_predicted_price` — fixed.
  ❌ MISSING: CORS settings blocked mobile app requests on some Android devices.
  ❌ MISSING: No request-ID or structured logging per request.
  ❌ MISSING: No input sanitisation on `farmer_location` field.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Add parent path so relative imports work
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_arbitrage.data_loader    import MandiDataLoader
from ml_arbitrage.price_predictor import PricePredictor
from ml_arbitrage.arbitrage_engine import (
    ArbitrageEngine, get_petrol_price, set_petrol_price,
)

logger = logging.getLogger("mandi_api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── Shared state ──────────────────────────────────────────────────────────────

_state: Dict = {
    "predictor":   None,
    "engine":      None,
    "df_featured": None,
    "startup_ok":  False,
    "startup_err": None,
}

DATASET_PATHS = [
    Path(__file__).parent.parent / "dataset" / "commodity_price.csv",
    Path(__file__).parent.parent / "data"    / "mandi_prices.csv",
]
VALID_CROPS = ["Onion", "Tomato", "Potato", "Wheat", "Cotton"]


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; clean up on shutdown."""
    logger.info("🚀 Mandi Intelligence API starting up …")
    try:
        loader    = MandiDataLoader()
        df        = None

        for path in DATASET_PATHS:
            if path.exists():
                loader.load_data(str(path))
                df = loader.filter_and_process(days=None)   # ALL historical data
                logger.info("Loaded real dataset: %s  (%d rows)", path.name, len(df))
                break

        if df is None or df.empty:
            logger.warning("No real dataset found — generating synthetic data.")
            df = loader.generate_rich_synthetic_data()

        predictor   = PricePredictor()
        df_featured = predictor.prepare_features(df)
        predictor.train_all_models(df_featured)

        engine = ArbitrageEngine(price_predictor=predictor)

        _state.update({
            "predictor":   predictor,
            "engine":      engine,
            "df_featured": df_featured,
            "startup_ok":  True,
        })
        logger.info("✅ Startup complete — %d mandis, %d crops",
                    df_featured["Mandi_Name"].nunique(),
                    df_featured["Crop"].nunique())

    except Exception as exc:
        _state["startup_err"] = str(exc)
        logger.exception("❌ Startup failed: %s", exc)

    yield  # ← API is live between yield and the next block

    logger.info("🛑 Shutting down Mandi Intelligence API.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Mandi Intelligence API",
    description = "ML-powered agricultural price predictions and profit-optimised recommendations.",
    version     = "3.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST", "OPTIONS"],
    allow_headers  = ["*"],
    expose_headers = ["X-Request-ID"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    crop:             str   = Field(..., description="Crop name (Onion/Tomato/Potato/Wheat/Cotton)")
    quantity:         float = Field(..., gt=0, le=100_000, description="Quantity in kg")
    farmer_location:  str   = Field("Gandhinagar", description="Farmer's city in Gujarat")
    petrol_price:     Optional[float] = Field(None, gt=0, le=500, description="Optional petrol price in Rs/L (uses system default if omitted)")

    @field_validator("crop")
    @classmethod
    def crop_must_be_valid(cls, v: str) -> str:
        for c in VALID_CROPS:
            if c.lower() == v.lower():
                return c
        raise ValueError(f"Invalid crop '{v}'. Choose from: {VALID_CROPS}")

    @field_validator("farmer_location")
    @classmethod
    def sanitise_location(cls, v: str) -> str:
        return v.strip()[:64]   # cap length, remove leading/trailing whitespace


class TransportDetail(BaseModel):
    vehicle:          str
    trips:            int = 1
    petrol_rate:      float = 94.37
    fuel:             float
    driver:           float
    toll:             float
    maintenance:      float
    loading_unloading:float


class CostBreakdown(BaseModel):
    gross_revenue:      float
    transport_cost:     float
    transport_detail:   TransportDetail
    storage_cost:       float
    perishability_cost: float
    qty_remaining_kg:   float


class OptimalStrategy(BaseModel):
    mandi:              str
    mandi_name:         str   # Frontend compatibility
    distance_km:        float
    price_per_kg:       float
    current_price:      float # Frontend compatibility
    days_to_wait:       int
    is_predicted_price: bool
    action_label:       str
    recommendation:     str   # Frontend compatibility (descriptive)
    net_profit:         float
    net_profit_lower:   float
    net_profit_upper:   float
    margin_pct:         float
    
    # Flattened for frontend
    gross_revenue:      float
    transport_cost:     float
    storage_cost:       float
    predicted_price_7d: Optional[float] = None
    
    cost_breakdown:     CostBreakdown


class RecommendResponse(BaseModel):
    request_id:          str
    crop:                str
    quantity_kg:         float
    quantity:            float # Frontend compatibility
    farmer_location:     str
    recommendation:      str
    summary:             str   # Frontend compatibility
    justification:       str
    optimal_strategy:    OptimalStrategy
    best_option:         OptimalStrategy # Frontend compatibility
    alternative_count:   int
    alternatives:        List[Dict]
    model_report_url:    str


class FeedbackRequest(BaseModel):
    farmer_id:     Optional[str]  = None
    mandi_name:    str
    crop:          str
    quantity:      float = Field(..., gt=0)
    actual_price:  float = Field(..., gt=0)
    sale_date:     str
    feedback:      Optional[str]  = None


# ── Middleware: request-ID injection ─────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid      = str(uuid.uuid4())[:8]
    start    = time.perf_counter()
    response = await call_next(request)
    ms       = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"]    = rid
    response.headers["X-Response-Time"] = f"{ms:.1f}ms"
    logger.info("[%s] %s %s → %d  (%.1f ms)", rid,
                request.method, request.url.path, response.status_code, ms)
    return response


# ── Helper ────────────────────────────────────────────────────────────────────

def _check_ready() -> None:
    if not _state["startup_ok"]:
        detail = _state.get("startup_err") or "System not ready — startup in progress."
        raise HTTPException(status_code=503, detail=detail)


def _generate_forecasts(crop: str) -> Optional[pd.DataFrame]:
    """Generate 7-day forecasts for all mandis for the given crop."""
    predictor   = _state["predictor"]
    df_featured = _state["df_featured"]
    if predictor is None or df_featured is None:
        return None
    try:
        forecasts = predictor.get_price_forecast_all_mandis(df_featured, crop, days_ahead=7)
        rows = []
        for mandi, fc_df in forecasts.items():
            for _, row in fc_df.iterrows():
                rows.append({
                    "Mandi_Name":      mandi,
                    "Crop":            crop,
                    "Day_Ahead":       int(row["Day_Ahead"]),
                    "Predicted_Price": float(row["Predicted_Price"]),
                    "Lower_Bound":     float(row.get("Lower_Bound", row["Predicted_Price"])),
                    "Upper_Bound":     float(row.get("Upper_Bound", row["Predicted_Price"])),
                })
        return pd.DataFrame(rows) if rows else None
    except Exception as exc:
        logger.warning("Forecast generation failed: %s", exc)
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Mandi Intelligence API v3.0",
        "status":  "ready" if _state["startup_ok"] else "loading",
        "docs":    "/docs",
    }


@app.get("/health")
async def health():
    df = _state.get("df_featured")
    return {
        "status":        "healthy" if _state["startup_ok"] else "degraded",
        "startup_error": _state.get("startup_err"),
        "models_loaded": _state["predictor"] is not None,
        "records":       len(df) if df is not None else 0,
        "mandis":        df["Mandi_Name"].nunique() if df is not None else 0,
        "crops":         df["Crop"].unique().tolist() if df is not None else [],
    }


@app.get("/mandis")
async def list_mandis():
    _check_ready()
    df = _state["df_featured"]
    result = []
    for mandi in df["Mandi_Name"].unique():
        md = df[df["Mandi_Name"] == mandi]
        result.append({
            "mandi_name":      mandi,
            "distance_km":     float(md["Distance_km"].iloc[0]),
            "available_crops": md["Crop"].unique().tolist(),
            "record_count":    len(md),
        })
    return {"total_mandis": len(result), "mandis": sorted(result, key=lambda x: x["distance_km"])}


@app.get("/model_report")
async def model_report():
    _check_ready()
    predictor = _state["predictor"]
    return predictor.get_model_report().to_dict(orient="records")


@app.get("/config/petrol")
async def get_petrol():
    """Return the current petrol price used for transport cost calculations."""
    return {"petrol_price_per_litre": get_petrol_price()}


@app.post("/config/petrol")
async def update_petrol(body: dict):
    """Update petrol price globally (affects all subsequent recommendations)."""
    price = body.get("price_per_litre")
    if not price or price <= 0 or price > 500:
        raise HTTPException(status_code=422, detail="price_per_litre must be between 0 and 500")
    set_petrol_price(float(price))
    return {
        "status": "updated",
        "petrol_price_per_litre": get_petrol_price(),
        "message": f"Petrol price set to Rs {price:.2f}/L for all future calculations.",
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get ML-based mandi recommendation for maximum net profit."""
    _check_ready()

    rid         = str(uuid.uuid4())[:8]
    df_featured = _state["df_featured"]
    engine      = _state["engine"]

    # Latest prices
    latest_date  = df_featured["Date"].max()
    df_current   = df_featured[df_featured["Date"] == latest_date]

    # Forecasts
    df_forecast  = _generate_forecasts(request.crop)

    try:
        result = engine.get_best_selling_strategy(
            current_qty_kg   = request.quantity,
            crop             = request.crop,
            current_location = request.farmer_location,
            df_current       = df_current,
            df_forecast      = df_forecast,
            petrol_price     = request.petrol_price,
        )
    except Exception as exc:
        logger.exception("Strategy generation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    opt = result["optimal_strategy"]
    bd  = opt["cost_breakdown"]
    td  = bd.get("transport_detail", {})

    # Compatibility: flatten alternative scenarios
    alternatives = []
    for sc in result.get("alternative_scenarios", []):
        s_bd = sc.get("cost_breakdown", {})
        alternatives.append({
            "mandi_name":     sc["mandi_name"], # Fix: engineer uses mandi_name in raw list
            "distance_km":    sc["distance_km"],
            "current_price":  sc["price_per_kg"],
            "net_profit":     sc["net_profit"],
            "recommendation": sc["action_label"],
            # flatten others if needed
        })

    # Find 7-day predicted price for hero section
    p_t7 = None
    if df_forecast is not None:
        f_mandi = df_forecast[df_forecast["Mandi_Name"] == opt["mandi"]]
        if not f_mandi.empty:
            p_t7 = float(f_mandi[f_mandi["Day_Ahead"] == 7]["Predicted_Price"].iloc[0])

    strategy = OptimalStrategy(
        mandi              = opt["mandi"],
        mandi_name         = opt["mandi"], # Alias
        distance_km        = opt["distance_km"],
        price_per_kg       = opt["price_per_kg"],
        current_price      = opt["price_per_kg"], # Alias
        days_to_wait       = opt["days_to_wait"],
        is_predicted_price = opt["is_predicted_price"],
        action_label       = opt["action_label"],
        recommendation     = result["recommendation"], # descriptive
        net_profit         = opt["net_profit"],
        net_profit_lower   = opt.get("net_profit_lower", opt["net_profit"]),
        net_profit_upper   = opt.get("net_profit_upper", opt["net_profit"]),
        margin_pct         = opt["margin_pct"],
        gross_revenue      = bd["gross_revenue"],
        transport_cost     = bd["transport_cost"],
        storage_cost       = bd["storage_cost"],
        predicted_price_7d = p_t7,
        cost_breakdown     = CostBreakdown(
            gross_revenue      = bd["gross_revenue"],
            transport_cost     = bd["transport_cost"],
            transport_detail   = TransportDetail(
                vehicle           = td.get("vehicle", "LCV"),
                fuel              = td.get("fuel", 0),
                driver            = td.get("driver", 0),
                toll              = td.get("toll", 0),
                maintenance       = td.get("maintenance", 0),
                loading_unloading = td.get("loading_unloading", 0),
            ),
            storage_cost       = bd["storage_cost"],
            perishability_cost = bd["perishability_cost"],
            qty_remaining_kg   = bd["qty_remaining_kg"],
        ),
    )

    return RecommendResponse(
        request_id       = rid,
        crop             = request.crop,
        quantity_kg      = request.quantity,
        quantity         = request.quantity, # Alias
        farmer_location  = request.farmer_location,
        recommendation   = result["recommendation"],
        summary          = result["recommendation"], # Alias
        justification    = result["justification"],
        optimal_strategy = strategy,
        best_option      = strategy, # Alias
        alternative_count = len(alternatives),
        alternatives      = alternatives,
        model_report_url  = "/model_report",
    )


# Backward-compat alias for /response
@app.post("/response")
async def response_alias(request: RecommendRequest):
    """Alias for /recommend (backward compatibility with v2 clients)."""
    return await recommend(request)


@app.post("/respond")
async def submit_feedback(request: FeedbackRequest):
    """Record actual sale data for model improvement."""
    farmer_id = request.farmer_id or f"FARMER_{abs(hash(request.mandi_name + request.sale_date)) % 9999:04d}"
    logger.info("Feedback [%s]: %s %s kg @ ₹%.2f on %s",
                farmer_id, request.crop, request.quantity,
                request.actual_price, request.sale_date)
    # TODO: persist to DB; trigger nightly retraining
    return {
        "status":    "success",
        "farmer_id": farmer_id,
        "message":   "Sale data recorded. Thank you — this improves recommendations for all farmers.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.ml_api:app", host="0.0.0.0", port=8000, reload=True)
