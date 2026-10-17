"""
arbitrage_engine.py — Net-profit optimiser: where (spatial) and when (temporal) to sell.

For every mandi m and every wait d = 0..7 days:

    qty_sold     = qty × exp(−λ·d)                      exponential spoilage
    gross        = qty_sold × price(m, d)
    transport    = fuel + driver + toll + maintenance + loading/unloading
                   (vehicle picked by load, multiple trips if needed)
    storage      = d × qty_quintal × ₹50
    net_profit   = gross − transport − storage

Spoilage is already priced in through `qty_sold`; `spoilage_cost` is reported
for transparency but not subtracted a second time.

Choosing a scenario
-------------------
Forecast prices are uncertain, today's observed price is not. Scenarios are
ranked by a risk-adjusted profit

    score = net_profit − RISK_AVERSION × (net_profit − net_profit_lower)

so the engine only recommends waiting when the expected gain outweighs a share
of the downside implied by the 80 % prediction interval.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .distance_calculator import calculate_distances_from_location

logger = logging.getLogger(__name__)

# ── Transport cost constants (March 2026) ────────────────────────────────────

_PETROL_PRICE_PER_LITRE: float = 94.0    # ₹/L (default, updatable at runtime)
NH_TOLL_PER_KM: float = 2.10             # ₹/km national highway
SH_TOLL_PER_KM: float = 0.70             # ₹/km state highway
NH_COVERAGE: float = 0.65                # share of Gujarat routes on NH
MAINTENANCE_PER_KM: float = 1.75         # ₹/km
LOADING_UNLOADING: float = 1_500.0       # ₹/trip
AVG_SPEED_KMPH: float = 45.0             # loaded truck
STORAGE_COST_PER_QUINTAL_DAY: float = 50.0
RISK_AVERSION: float = 0.5
MAX_WAIT_DAYS: int = 7
LOCAL_RADIUS_KM: float = 30.0

# Vehicle profiles: (fuel efficiency km/L, driver ₹/day, payload quintal)
_VEHICLES = {
    "MINI_TRUCK":   (16.0, 700,   7.5),
    "LCV":          (12.0, 1_100, 25.0),
    "MEDIUM_TRUCK": (7.0,  1_100, 60.0),
    "HEAVY_TRUCK":  (4.0,  1_900, 180.0),
}

PERISHABILITY = {"Onion": 0.030, "Tomato": 0.055, "Potato": 0.020}   # λ per day


def get_petrol_price() -> float:
    return _PETROL_PRICE_PER_LITRE


def set_petrol_price(price_per_litre: float) -> None:
    global _PETROL_PRICE_PER_LITRE
    if price_per_litre <= 0:
        raise ValueError("Petrol price must be positive.")
    _PETROL_PRICE_PER_LITRE = price_per_litre
    logger.info("Petrol price updated to ₹%.2f/L", price_per_litre)


def _select_vehicle(qty_quintal: float) -> Tuple[str, float, float, float]:
    """Smallest vehicle that carries the load in one trip, else the heavy truck."""
    for name, (eff, wage, cap) in _VEHICLES.items():
        if qty_quintal <= cap:
            return name, eff, float(wage), cap
    eff, wage, cap = _VEHICLES["HEAVY_TRUCK"]
    return "HEAVY_TRUCK", eff, float(wage), cap


def transport_cost(
    distance_km: float,
    qty_quintal: float,
    petrol_price: Optional[float] = None,
) -> Tuple[float, Dict]:
    """Five-component trip cost, scaled by the number of trips needed."""
    if distance_km <= 0:
        return 0.0, {"vehicle": "NONE", "trips": 0, "petrol_rate": 0.0, "fuel": 0.0,
                     "driver": 0.0, "toll": 0.0, "maintenance": 0.0, "loading_unloading": 0.0}

    petrol = petrol_price if petrol_price and petrol_price > 0 else _PETROL_PRICE_PER_LITRE
    vehicle, eff, wage, payload = _select_vehicle(qty_quintal)
    trips = max(1, math.ceil(qty_quintal / payload))

    round_trip_hours = (distance_km / AVG_SPEED_KMPH) * 2.0 + 1.5
    blended_toll = NH_COVERAGE * NH_TOLL_PER_KM + (1 - NH_COVERAGE) * SH_TOLL_PER_KM
    parts = {
        "fuel": distance_km / eff * petrol * trips,
        "driver": round_trip_hours * (wage / 8.0) * trips,
        "toll": distance_km * blended_toll * trips,
        "maintenance": distance_km * MAINTENANCE_PER_KM * trips,
        "loading_unloading": LOADING_UNLOADING * trips,
    }
    total = sum(parts.values())
    detail = {"vehicle": vehicle, "trips": trips, "petrol_rate": round(petrol, 2),
              **{k: round(v, 2) for k, v in parts.items()}}
    return round(total, 2), detail


class ArbitrageEngine:
    """Turns a price forecast table into a sell recommendation."""

    def get_best_selling_strategy(
        self,
        current_qty_kg: float,
        crop: str,
        forecast: pd.DataFrame,
        current_location: str = "Gandhinagar",
        crop_perishability_factor: Optional[float] = None,
        petrol_price: Optional[float] = None,
    ) -> Dict:
        """
        Parameters
        ----------
        forecast : PriceForecaster.forecast() output for `crop` — one row per
                   (Mandi_Name, Day_Ahead) with Predicted/Lower/Upper prices and
                   Is_Observed (True for prices actually reported today).
        """
        if forecast is None or forecast.empty:
            return self._error(f"No price data available for {crop}.")

        qty_quintal = current_qty_kg / 100.0
        lam = crop_perishability_factor or PERISHABILITY.get(crop, 0.02)
        distances = calculate_distances_from_location(current_location,
                                                      forecast["Mandi_Name"].unique().tolist())
        # Transport does not depend on the day, so price each mandi once.
        trips = {m: transport_cost(d, qty_quintal, petrol_price) for m, d in distances.items()}

        scenarios = []
        for row in forecast[forecast["Day_Ahead"] <= MAX_WAIT_DAYS].itertuples(index=False):
            mandi = row.Mandi_Name
            scenarios.append(self._scenario(
                mandi=mandi, distance_km=distances[mandi], qty_quintal=qty_quintal,
                wait_days=int(row.Day_Ahead), lam=lam, price=float(row.Predicted_Price),
                price_lower=float(row.Lower_Bound), price_upper=float(row.Upper_Bound),
                is_forecast=not bool(row.Is_Observed), transport=trips[mandi],
            ))

        scenarios.sort(key=lambda s: s["score"], reverse=True)
        optimal = scenarios[0]
        same_day = [s for s in scenarios if s["days_to_wait"] == 0]
        best_today = same_day[0] if same_day else None
        today_here = next((s for s in same_day if s["mandi_name"] == optimal["mandi_name"]), None)

        # Alternatives: the best plan at each *other* mandi
        seen = {optimal["mandi_name"]}
        alternatives = []
        for s in scenarios[1:]:
            if s["mandi_name"] not in seen:
                seen.add(s["mandi_name"])
                alternatives.append(s)

        return self._build_response(optimal, best_today, today_here, alternatives,
                                    current_qty_kg, qty_quintal, crop, current_location)

    # ── Scenario economics ───────────────────────────────────────────────────

    @staticmethod
    def _scenario(mandi: str, distance_km: float, qty_quintal: float, wait_days: int,
                  lam: float, price: float, price_lower: float, price_upper: float,
                  is_forecast: bool, transport: Tuple[float, Dict]) -> Dict:
        qty_kg = qty_quintal * 100
        qty_sold_kg = qty_kg * math.exp(-lam * wait_days)
        gross = qty_sold_kg * price
        transport_total, transport_detail = transport
        storage = wait_days * qty_quintal * STORAGE_COST_PER_QUINTAL_DAY
        spoilage_value = (qty_kg - qty_sold_kg) * price          # informational only
        net = gross - transport_total - storage
        net_lower = net - (price - price_lower) * qty_sold_kg
        net_upper = net + (price_upper - price) * qty_sold_kg

        if wait_days > 0:
            label = "STORE_FOR_T_DAYS"
        elif distance_km < LOCAL_RADIUS_KM:
            label = "SELL_LOCAL_NOW"
        else:
            label = "TRAVEL_TO_FAR_MANDI"

        return {
            "mandi_name": mandi,
            "distance_km": round(distance_km, 1),
            "price_per_kg": round(price, 2),
            "days_to_wait": wait_days,
            "is_forecast": is_forecast,
            "action_label": label,
            "qty_remaining_kg": round(qty_sold_kg, 2),
            "gross_revenue": round(gross, 2),
            "transport_cost": transport_total,
            "transport_detail": transport_detail,
            "storage_cost": round(storage, 2),
            "spoilage_cost": round(spoilage_value, 2),
            "net_profit": round(net, 2),
            "net_profit_lower": round(net_lower, 2),
            "net_profit_upper": round(net_upper, 2),
            "margin_pct": round(net / gross * 100, 2) if gross else 0.0,
            "score": net - RISK_AVERSION * (net - net_lower),
        }

    # ── Response ─────────────────────────────────────────────────────────────

    def _build_response(self, optimal: Dict, best_today: Optional[Dict],
                        today_here: Optional[Dict], alternatives: List[Dict],
                        qty_kg: float, qty_quintal: float, crop: str, location: str) -> Dict:
        mandi, days = optimal["mandi_name"], optimal["days_to_wait"]
        action = f"Sell at {mandi} TODAY" if days == 0 else f"Wait {days} days, then sell at {mandi}"
        return {
            "recommendation": action,
            "strategy_type": "SPATIAL" if days == 0 else "TEMPORAL_SPATIAL",
            "crop": crop,
            "farmer_location": location,
            "quantity_kg": qty_kg,
            "quantity_quintal": round(qty_quintal, 2),
            "optimal_strategy": {
                "mandi": mandi,
                "distance_km": optimal["distance_km"],
                "price_per_kg": optimal["price_per_kg"],
                "days_to_wait": days,
                "is_predicted_price": optimal["is_forecast"],
                "action_label": optimal["action_label"],
                "net_profit": optimal["net_profit"],
                "net_profit_lower": optimal["net_profit_lower"],
                "net_profit_upper": optimal["net_profit_upper"],
                "margin_pct": optimal["margin_pct"],
                "cost_breakdown": {
                    "gross_revenue": optimal["gross_revenue"],
                    "transport_cost": optimal["transport_cost"],
                    "transport_detail": optimal["transport_detail"],
                    "storage_cost": optimal["storage_cost"],
                    "perishability_cost": optimal["spoilage_cost"],
                    "qty_remaining_kg": optimal["qty_remaining_kg"],
                },
            },
            "justification": self._justify(optimal, best_today, today_here, qty_kg, crop),
            "alternative_scenarios": alternatives[:4],
        }

    @staticmethod
    def _justify(opt: Dict, best_today: Optional[Dict], today_here: Optional[Dict],
                 qty_kg: float, crop: str) -> str:
        mandi, days, price = opt["mandi_name"], opt["days_to_wait"], opt["price_per_kg"]
        td = opt["transport_detail"]
        price_kind = "forecast price" if opt["is_forecast"] else "today's price"
        if days == 0:
            return (f"Sell at {mandi} ({opt['distance_km']:.0f} km) now at {price_kind} "
                    f"₹{price:.2f}/kg. After ₹{opt['transport_cost']:,.0f} transport "
                    f"(fuel ₹{td.get('fuel', 0):,.0f} + driver ₹{td.get('driver', 0):,.0f} "
                    f"+ tolls + maintenance + loading), net profit is ₹{opt['net_profit']:,.0f} "
                    f"for {qty_kg:,.0f} kg of {crop}.")
        ref_price = today_here["price_per_kg"] if today_here else price
        gain = opt["net_profit"] - (best_today["net_profit"] if best_today else 0)
        return (f"Hold for {days} days, then sell at {mandi}. The {crop} price there is "
                f"forecast to move from ₹{ref_price:.2f} to ₹{price:.2f}/kg "
                f"(80% range ₹{opt['net_profit_lower']:,.0f}–₹{opt['net_profit_upper']:,.0f} net). "
                f"After ₹{opt['storage_cost']:,.0f} storage and ₹{opt['spoilage_cost']:,.0f} "
                f"of spoilage, that is ₹{gain:,.0f} more than the best sale today. "
                f"Expected net profit: ₹{opt['net_profit']:,.0f}.")

    @staticmethod
    def _error(msg: str) -> Dict:
        logger.error(msg)
        return {"error": msg, "recommendation": "Unable to generate recommendation."}
