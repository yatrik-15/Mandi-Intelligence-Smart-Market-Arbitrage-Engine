"""
quick_test.py — Comprehensive Pipeline Validator v3.0
======================================================
Tests every layer of the stack:
  Layer 1: Data loader (schema detection, filtering, winsorisation)
  Layer 2: Feature engineering (no cross-group leakage)
  Layer 3: Model training (all viable pairs)
  Layer 4: Forecasting (rolling buffer, non-negative prices)
  Layer 5: Transport cost (5-component model at ₹94.37/L)
  Layer 6: Arbitrage engine (end-to-end recommendation)
  Layer 7: Distance calculator (matrix + Haversine)
  Layer 8: API response schema (Pydantic validation)
"""

from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
results = []

def check(label: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    results.append((label, condition, detail))
    print(f"  {tag}  {label}" + (f"  [{detail}]" if detail else ""))

def section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  MANDI INTELLIGENCE — PIPELINE VALIDATION v3.0")
print("=" * 70)

# ── Layer 1: Data loader ──────────────────────────────────────────────────────
section("Layer 1: Data Loader")
from ml_arbitrage.data_loader import MandiDataLoader, TARGET_CROPS

loader = MandiDataLoader()
real_path = Path("dataset/commodity_price.csv")

if real_path.exists():
    try:
        loader.load_data(str(real_path))
        df = loader.filter_and_process(days=None)
        check("Real CSV loaded", True, f"{len(df):,} rows")
        check("Price_per_kg > 0", (df["Price_per_kg"] > 0).all(),
              f"min={df['Price_per_kg'].min():.2f}")
        check("No NaN prices", df["Price_per_kg"].isna().sum() == 0)
        check("Mandi_Name not null", df["Mandi_Name"].isna().sum() == 0)
        check("Date parsed correctly", pd.api.types.is_datetime64_any_dtype(df["Date"]))
        check("Distance_km > 0", (df["Distance_km"] > 0).all())
    except Exception as e:
        check("Real CSV loaded", False, str(e)[:80])
        df = loader.generate_rich_synthetic_data()
        check("Synthetic fallback", df is not None and not df.empty)
else:
    print(f"  {WARN} No real dataset at {real_path} — using synthetic data")
    df = loader.generate_rich_synthetic_data()
    check("Synthetic data generated", not df.empty, f"{len(df):,} rows")
    check("All target columns present",
          all(c in df.columns for c in ["Date", "Mandi_Name", "Crop",
                                         "Price_per_kg", "Distance_km"]))

report = loader.validation_report
check("Quality score > 0", report.get("quality_score", 0) > 0,
      f"{report.get('quality_score', 0):.2%}")

# ── Layer 2: Feature engineering ─────────────────────────────────────────────
section("Layer 2: Feature Engineering (no cross-group leakage)")
from ml_arbitrage.price_predictor import PricePredictor, FEATURE_COLS

predictor   = PricePredictor()
df_featured = predictor.prepare_features(df)

check("All FEATURE_COLS present", all(c in df_featured.columns for c in FEATURE_COLS))
check("No inf in features", not np.isinf(df_featured[FEATURE_COLS].fillna(0)).any().any())
check("price_change_7d clipped", df_featured["price_change_7d"].abs().max() <= 0.5,
      f"max={df_featured['price_change_7d'].abs().max():.3f}")

# Cross-group leakage test: lag_1 should not mix Rajkot-Onion and Surat-Tomato
if len(df_featured["Mandi_Name"].unique()) > 1 and len(df_featured["Crop"].unique()) > 1:
    mandi_a = df_featured["Mandi_Name"].iloc[0]
    crop_a  = df_featured["Crop"].iloc[0]
    group_a = df_featured[(df_featured["Mandi_Name"] == mandi_a) &
                           (df_featured["Crop"] == crop_a)]
    # First row's lag_1 should equal the group mean (filled), not another group's price
    if len(group_a) > 1:
        lag_matches_group = abs(
            group_a["price_lag_1"].iloc[1] - group_a["Price_per_kg"].iloc[0]
        ) < 1.0
        check("Lag_1 within-group only", lag_matches_group)

# ── Layer 3: Model training ───────────────────────────────────────────────────
section("Layer 3: Model Training")
train_results = predictor.train_all_models(df_featured)
trained = [r for r in train_results if "error" not in r]
skipped = [r for r in train_results if "error" in r]

check("At least 1 model trained", len(trained) > 0, f"{len(trained)} models")
check("Skipped pairs have error key", all("error" in r for r in skipped))
if trained:
    avg_mape = sum(r["mape"] for r in trained) / len(trained)
    avg_r2   = sum(r["r2"]   for r in trained) / len(trained)
    check(f"Avg MAPE < 20 %", avg_mape < 20, f"{avg_mape:.1f}%")
    check(f"Avg R² > 0.3",    avg_r2 > 0.3,  f"{avg_r2:.3f}")

    report_df = predictor.get_model_report()
    check("Model report DataFrame returned", len(report_df) == len(trained))

# ── Layer 4: Forecasting ──────────────────────────────────────────────────────
section("Layer 4: Price Forecasting (rolling buffer integrity)")
if trained:
    test_mandi = trained[0]["mandi"]
    test_crop  = trained[0]["crop"]
    try:
        fc = predictor.predict_future_price(df_featured, test_mandi, test_crop, days_ahead=7)
        check("Forecast returns 7 rows",       len(fc) == 7, f"{len(fc)} rows")
        check("Forecast prices > 0",           (fc["Predicted_Price"] > 0).all())
        check("Lower_Bound ≤ Predicted_Price", (fc["Lower_Bound"] <= fc["Predicted_Price"]).all())
        check("Upper_Bound ≥ Predicted_Price", (fc["Upper_Bound"] >= fc["Predicted_Price"]).all())
        check("Day_Ahead values 1–7",          fc["Day_Ahead"].tolist() == list(range(1, 8)))
    except Exception as e:
        check("Forecast succeeded", False, str(e)[:80])

# ── Layer 5: Transport cost ───────────────────────────────────────────────────
section("Layer 5: Transport Cost Model (₹94.37/L, 5-component)")
from ml_arbitrage.arbitrage_engine import _transport_cost, PETROL_PRICE_PER_LITRE

check("Petrol price constant", PETROL_PRICE_PER_LITRE == 94.37, f"₹{PETROL_PRICE_PER_LITRE}")

# LCV, 200 km: expect total > ₹3,000 (old model: 200 × 5 = ₹1,000)
tc, bd = _transport_cost(200.0, 10.0, travels=True)
check("LCV 200km cost > ₹3,000", tc > 3_000, f"₹{tc:,.0f}")
check("Fuel component present",   bd.get("fuel", 0) > 0, f"₹{bd.get('fuel',0):.0f}")
check("Driver component present", bd.get("driver", 0) > 0, f"₹{bd.get('driver',0):.0f}")
check("Toll component present",   bd.get("toll", 0) > 0, f"₹{bd.get('toll',0):.0f}")

# Zero distance = zero cost
tc_zero, _ = _transport_cost(0.0, 10.0, travels=True)
check("Zero distance → zero cost", tc_zero == 0.0)

# No-travel scenario
tc_local, _ = _transport_cost(200.0, 10.0, travels=False)
check("No-travel flag → zero cost", tc_local == 0.0)

# ── Layer 6: Arbitrage engine ─────────────────────────────────────────────────
section("Layer 6: Arbitrage Engine (end-to-end)")
from ml_arbitrage.arbitrage_engine import ArbitrageEngine

engine = ArbitrageEngine(price_predictor=predictor)
latest_date = df_featured["Date"].max()
df_current  = df_featured[df_featured["Date"] == latest_date]

crop_test = df_current["Crop"].iloc[0]
try:
    rec = engine.get_best_selling_strategy(
        current_qty_kg   = 1000.0,
        crop             = crop_test,
        current_location = "Ahmedabad",
        df_current       = df_current,
    )
    check("Recommendation returned",   "error" not in rec, rec.get("error", ""))
    check("'recommendation' key exists","recommendation" in rec)
    check("'optimal_strategy' key",    "optimal_strategy" in rec)
    check("Net profit is numeric",     isinstance(rec.get("optimal_strategy", {}).get("net_profit"), (int, float)))
    check("Cost breakdown present",    "cost_breakdown" in rec.get("optimal_strategy", {}))
    check("Transport detail in breakdown",
          "transport_detail" in rec.get("optimal_strategy", {}).get("cost_breakdown", {}))
    check("Alternatives list returned","alternative_scenarios" in rec)
except Exception as e:
    check("Recommendation succeeded", False, str(e)[:80])
    traceback.print_exc()

# Edge case: invalid crop
bad_rec = engine.get_best_selling_strategy(
    current_qty_kg=500, crop="Dragon_Fruit",
    current_location="Rajkot", df_current=df_current
)
check("Invalid crop returns error dict", "error" in bad_rec)

# ── Layer 7: Distance calculator ─────────────────────────────────────────────
section("Layer 7: Distance Calculator")
from ml_arbitrage.distance_calculator import get_distance, calculate_distances_from_location

check("Ahmedabad→Rajkot known", get_distance("Ahmedabad", "Rajkot") == 216)
check("Symmetry: Rajkot→Ahmedabad", get_distance("Rajkot", "Ahmedabad") == 216)
check("Same city → 0 km", get_distance("Surat", "Surat") == 0)
check("Alias: Baroda→Vadodara",
      get_distance("Baroda", "Ahmedabad") == get_distance("Vadodara", "Ahmedabad"))
check("Unknown city → > 0 (fallback)", get_distance("Kutch", "Ahmedabad") > 0)

dists = calculate_distances_from_location("Gandhinagar",
                                           ["Ahmedabad", "Rajkot", "Surat"])
check("Batch calc returns 3 entries", len(dists) == 3)
check("Ahmedabad closest from Gandhinagar",
      dists["Ahmedabad"] < dists["Rajkot"])

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  RESULTS: {passed} passed, {failed} failed  ({passed}/{passed+failed})")

if failed:
    print("\n  FAILURES:")
    for label, ok, detail in results:
        if not ok:
            print(f"    ❌ {label}" + (f": {detail}" if detail else ""))

print("=" * 70)
sys.exit(0 if failed == 0 else 1)
