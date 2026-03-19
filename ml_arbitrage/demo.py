"""
demo.py — End-to-End Integration Demo
======================================
Generates synthetic Mandi data, trains the full three-stage ensemble,
runs batch predictions, and prints a complete evaluation report.

Run:
    python -m mandi_arbitrage_engine.demo
"""

from __future__ import annotations

import logging
import warnings
from datetime import timedelta, date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MandiDemo")

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generator
# ──────────────────────────────────────────────────────────────────────────────

CROPS     = ["wheat", "onion", "tomato", "soybean", "rice"]
WEATHERS  = ["clear", "rainy", "cloudy", "foggy"]
CLASSES   = ["SELL_LOCAL_NOW", "TRAVEL_TO_FAR_MANDI", "STORE_FOR_T_DAYS"]

RNG = np.random.default_rng(42)


def _make_price_series(n_days: int = 365, base: float = 1800.0) -> pd.DataFrame:
    """Synthetic price series with trend, seasonality, and noise."""
    dates  = [date(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    trend  = np.linspace(0, 200, n_days)
    season = 150 * np.sin(2 * np.pi * np.arange(n_days) / 182.5)
    noise  = RNG.normal(0, 40, n_days)
    prices = np.maximum(base + trend + season + noise, 100)

    # Inject 5 % missing values
    mask   = RNG.random(n_days) < 0.05
    prices[mask] = np.nan

    return pd.DataFrame({"ds": dates, "y": prices})


def make_synthetic_records(n: int = 40):
    """Generate n synthetic MandiRecord objects."""
    # Import here to avoid circular issues at module level in the demo
    from mandi_arbitrage_engine.engine import MandiRecord

    records = []
    for i in range(n):
        crop     = CROPS[i % len(CROPS)]
        base_px  = RNG.uniform(1200, 3000)
        record   = MandiRecord(
            mandi_id                 = f"MANDI_{i:04d}",
            crop_name                = crop,
            lat                      = RNG.uniform(18.0, 32.0),
            lon                      = RNG.uniform(72.0, 88.0),
            price_series             = _make_price_series(base=base_px),
            weather_condition        = WEATHERS[i % len(WEATHERS)],
            distance_km              = RNG.uniform(10, 500),
            fuel_cost_per_km         = RNG.uniform(4.0, 8.0),
            perishability_decay_rate = RNG.uniform(0.01, 0.12),
            storage_cost_per_day     = RNG.uniform(20, 120),
            qty_available            = RNG.uniform(10, 200),
        )
        records.append(record)
    return records


def make_synthetic_labels(n: int) -> pd.Series:
    """Random labels for demo — in production these come from historical outcomes."""
    return pd.Series(RNG.choice(CLASSES, size=n), name="action")


# ──────────────────────────────────────────────────────────────────────────────
# Main Demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    from mandi_arbitrage_engine.engine import MandiArbitrageEngine
    from mandi_arbitrage_engine.config import EngineConfig

    logger.info("Generating %d synthetic Mandi records …", 40)
    all_records = make_synthetic_records(40)
    all_labels  = make_synthetic_labels(40)

    train_records, test_records = all_records[:32], all_records[32:]
    train_labels,  test_labels  = all_labels[:32],  all_labels[32:]

    # ── Instantiate engine ────────────────────────────────────────────────────
    engine = MandiArbitrageEngine(config=EngineConfig(), model_dir="models_demo/")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("═══ TRAINING ═══")
    engine.train(train_records, train_labels)

    # ── Predict single ────────────────────────────────────────────────────────
    logger.info("═══ SINGLE PREDICTION ═══")
    decision = engine.predict(test_records[0])
    print("\n" + "═" * 68)
    print("  SINGLE PREDICTION RESULT")
    print("═" * 68)
    print(f"  Mandi ID   : {decision.mandi_id}")
    print(f"  Crop       : {decision.crop_name}")
    print(f"  ACTION     : {decision.action}")
    print(f"  Confidence : {decision.confidence:.1%}")
    print(f"  Price t+3  : ₹{decision.price_t3:.2f}")
    print(f"  Price t+7  : ₹{decision.price_t7:.2f}")
    print(f"  Net Profit t+3: ₹{decision.net_profit_t3:.2f}")
    print(f"  Net Profit t+7: ₹{decision.net_profit_t7:.2f}")
    print(f"\n  Reasoning  : {decision.reasoning}")
    print("═" * 68)

    # ── Batch predict ─────────────────────────────────────────────────────────
    logger.info("═══ BATCH PREDICTION ═══")
    decisions = engine.predict_batch(test_records)
    batch_df  = pd.DataFrame([{
        "mandi_id":      d.mandi_id,
        "crop_name":     d.crop_name,
        "action":        d.action,
        "confidence":    d.confidence,
        "price_t3":      d.price_t3,
        "price_t7":      d.price_t7,
        "net_profit_t3": d.net_profit_t3,
        "net_profit_t7": d.net_profit_t7,
    } for d in decisions])
    print("\nBatch Predictions:")
    print(batch_df.to_string(index=False))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("═══ EVALUATION ═══")
    metrics = engine.evaluate(test_records, test_labels)

    # Action distribution
    print("\nAction distribution in test set:")
    print(batch_df["action"].value_counts().to_string())

    logger.info("Demo complete.")


if __name__ == "__main__":
    run_demo()
