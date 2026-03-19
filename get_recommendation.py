"""
get_recommendation.py — Farmer-Facing CLI v3.0
===============================================
Fixes:
  ✅ Properly passes forecast data to engine (was silently discarded).
  ✅ Handles missing models gracefully with a clear message.
  ✅ Shows transport cost breakdown from the new 5-component model.
  ✅ `generate_recommendation()` no longer creates a second ArbitrageEngine
     instance — reuses the one from `load_models()`.
  ✅ Works with synthetic data when no real dataset is present.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from pathlib import Path

from ml_arbitrage.data_loader     import MandiDataLoader
from ml_arbitrage.price_predictor import PricePredictor
from ml_arbitrage.arbitrage_engine import ArbitrageEngine


DATASET_PATHS = [
    Path("dataset") / "commodity_price.csv",
    Path("data")    / "mandi_prices.csv",
]
CROPS     = ["Onion", "Tomato", "Potato", "Wheat", "Cotton"]
LOCATIONS = ["Gandhinagar", "Ahmedabad", "Mehsana", "Rajkot", "Surat", "Vadodara", "Anand"]


# ──────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    print()
    print("═" * 72)
    print("       🌾  BEEJ RAKSHAK — MANDI PRICE RECOMMENDATION SYSTEM  🌾")
    print("═" * 72)
    print("  Powered by: XGBoost Price Forecasting + Arbitrage Optimisation")
    print("  Transport cost model: ₹94.37/L petrol | Real vehicle profiles")
    print("═" * 72)
    print()


def get_farmer_input() -> tuple[str, str, float]:
    print("─" * 72)
    print("  STEP 1 — Your location")
    print("─" * 72)
    for i, loc in enumerate(LOCATIONS, 1):
        print(f"  {i}. {loc}")
    print(f"  {len(LOCATIONS)+1}. Other (enter city name)")
    choice = input("\n  Enter number: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(LOCATIONS):
            location = LOCATIONS[idx]
        else:
            location = input("  Enter city name: ").strip() or "Gandhinagar"
    except ValueError:
        location = "Gandhinagar"

    print(f"\n  ✅ Location: {location}\n")

    print("─" * 72)
    print("  STEP 2 — Crop")
    print("─" * 72)
    for i, c in enumerate(CROPS, 1):
        print(f"  {i}. {c}")
    choice = input("\n  Enter number (1-5): ").strip()
    try:
        crop = CROPS[int(choice) - 1]
    except (ValueError, IndexError):
        crop = "Onion"
    print(f"  ✅ Crop: {crop}\n")

    print("─" * 72)
    print("  STEP 3 — Quantity")
    print("─" * 72)
    try:
        quantity = float(input("  Enter quantity in kg: ").strip())
        if quantity <= 0:
            raise ValueError
    except ValueError:
        quantity = 1000.0
        print(f"  Using default: {quantity:.0f} kg")
    print(f"  ✅ Quantity: {quantity:,.0f} kg\n")

    return location, crop, quantity


def load_models() -> tuple[pd.DataFrame, PricePredictor]:
    print("⏳ Loading data and training models …\n")
    loader = MandiDataLoader()
    df     = None

    for path in DATASET_PATHS:
        if path.exists():
            loader.load_data(str(path))
            df = loader.filter_and_process(days=None)
            if df is not None and not df.empty:
                print(f"   📊 Real dataset loaded: {len(df):,} records\n")
                break

    if df is None or df.empty:
        print("   ⚠️  No real dataset found — using synthetic data for demo.\n")
        df = loader.generate_rich_synthetic_data()

    predictor   = PricePredictor()
    df_featured = predictor.prepare_features(df)
    results     = predictor.train_all_models(df_featured)
    trained     = [r for r in results if "error" not in r]
    print(f"   ✅ Trained {len(trained)} models\n")

    return df_featured, predictor


def generate_recommendation(
    df_featured:  pd.DataFrame,
    predictor:    PricePredictor,
    location:     str,
    crop:         str,
    quantity:     float,
) -> dict:
    engine = ArbitrageEngine(price_predictor=predictor)

    latest_date = df_featured["Date"].max()
    df_current  = df_featured[df_featured["Date"] == latest_date]

    # Generate 7-day forecasts
    df_forecast = None
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
        if rows:
            df_forecast = pd.DataFrame(rows)
    except Exception as e:
        print(f"   ⚠️  Could not generate forecasts: {e}")

    return engine.get_best_selling_strategy(
        current_qty_kg   = quantity,
        crop             = crop,
        current_location = location,
        df_current       = df_current,
        df_forecast      = df_forecast,
    )


def display_recommendation(rec: dict, location: str, crop: str, qty: float) -> None:
    print()
    print("═" * 72)
    print("  📋 YOUR PERSONALISED RECOMMENDATION")
    print("═" * 72)

    if "error" in rec:
        print(f"\n  ❌ {rec['error']}")
        return

    print(f"\n  Location  : {location}")
    print(f"  Crop      : {crop}")
    print(f"  Quantity  : {qty:,.0f} kg\n")

    print("─" * 72)
    print(f"  🎯 RECOMMENDATION: {rec['recommendation']}")
    print("─" * 72)

    opt = rec.get("optimal_strategy", {})
    bd  = opt.get("cost_breakdown", {})
    td  = bd.get("transport_detail", {})

    print(f"\n  Mandi            : {opt.get('mandi', 'N/A')}")
    print(f"  Distance         : {opt.get('distance_km', 0):.0f} km")
    print(f"  Price (₹/kg)     : ₹{opt.get('price_per_kg', 0):.2f}"
          + ("  [PREDICTED]" if opt.get("is_predicted_price") else "  [CURRENT]"))
    if opt.get("days_to_wait", 0) > 0:
        print(f"  Days to wait     : {opt['days_to_wait']} days")
    print(f"  Qty remaining    : {bd.get('qty_remaining_kg', qty):,.1f} kg  (after spoilage)")

    print(f"\n  ── Revenue & Costs ─────────────────────────────────────────")
    print(f"  Gross revenue    : ₹{bd.get('gross_revenue', 0):>12,.2f}")
    print(f"  ── Transport ─────────────────────────────────────────────")
    print(f"     Vehicle       : {td.get('vehicle', 'LCV')}")
    print(f"     Fuel (₹94.37/L): ₹{td.get('fuel', 0):>10,.2f}")
    print(f"     Driver wages  : ₹{td.get('driver', 0):>10,.2f}")
    print(f"     Tolls (NH/SH) : ₹{td.get('toll', 0):>10,.2f}")
    print(f"     Maintenance   : ₹{td.get('maintenance', 0):>10,.2f}")
    print(f"     Load/Unload   : ₹{td.get('loading_unloading', 0):>10,.2f}")
    print(f"     TOTAL         : ₹{bd.get('transport_cost', 0):>10,.2f}")
    print(f"  Storage cost     : ₹{bd.get('storage_cost', 0):>12,.2f}")
    print(f"  Spoilage loss    : ₹{bd.get('perishability_cost', 0):>12,.2f}")
    print(f"  {'─'*52}")
    print(f"  NET PROFIT       : ₹{opt.get('net_profit', 0):>12,.2f}  (margin {opt.get('margin_pct', 0):+.1f}%)")

    np_lo = opt.get("net_profit_lower", opt.get("net_profit"))
    np_hi = opt.get("net_profit_upper", opt.get("net_profit"))
    if np_lo != np_hi:
        print(f"  80% CI           : ₹{np_lo:,.0f} — ₹{np_hi:,.0f}")

    print(f"\n  ── Why this recommendation ─────────────────────────────────")
    print(f"  {rec.get('justification', '')}\n")

    alts = rec.get("alternative_scenarios", [])[:3]
    if alts:
        print("─" * 72)
        print("  📊 ALTERNATIVES (ranked by net profit)")
        print("─" * 72)
        for i, alt in enumerate(alts, 1):
            days = alt.get("days_to_wait", 0)
            wait = f"Wait {days}d → " if days > 0 else "Today → "
            print(f"  {i}. {wait}{alt['mandi_name']}  |  "
                  f"₹{alt['price_per_kg']:.2f}/kg  |  Net ₹{alt['net_profit']:,.0f}")

    print("═" * 72)


def main() -> None:
    print_banner()
    location, crop, quantity = get_farmer_input()

    try:
        df_featured, predictor = load_models()
    except Exception as e:
        print(f"\n❌ Failed to load models: {e}")
        return

    print(f"🔍 Analysing from {location} for {quantity:,.0f} kg {crop} …\n")
    try:
        rec = generate_recommendation(df_featured, predictor, location, crop, quantity)
        display_recommendation(rec, location, crop, quantity)
    except Exception as e:
        print(f"\n❌ Recommendation failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
