"""
train.py — Retrain the forecasters and write the evaluation report.

    python scripts/train.py                              # dataset/commodity_price.csv
    MANDI_DATASET=dataset/agmarknet_gujarat.csv python scripts/train.py

Outputs
    models/<Crop>.ubj, models/<Crop>.meta.json   (loaded by the API at startup)
    reports/MODEL_REPORT.md                       (honest held-out metrics)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_arbitrage.service import MandiService, dataset_path  # noqa: E402

REPORT = ROOT / "reports" / "MODEL_REPORT.md"


def _row(name: str, m: dict, base: dict) -> str:
    skill = (1 - m["mae"] / base["mae"]) * 100
    return (f"| {name} | ₹{m['mae']:.2f} | ₹{m['rmse']:.2f} | {m['mape']:.2f}% "
            f"| {m['r2']:.3f} | {skill:+.0f}% |")


def write_report(svc: MandiService, seconds: float) -> str:
    lines = [
        "# Model Report",
        "",
        f"Dataset: `{dataset_path().name}` — {len(svc.panel):,} mandi-days, "
        f"{svc.panel['Mandi_Name'].nunique()} mandis, "
        f"{svc.panel['Date'].min().date()} → {svc.panel['Date'].max().date()}. "
        f"Training took {seconds:.0f}s.",
        "",
        "All numbers are on the held-out **test** period (the last 20% of dates), which the "
        "model never saw during training or early stopping. *Skill* is the MAE reduction "
        "versus persistence (\"the price stays where it is today\").",
        "",
    ]
    for crop in svc.crops:
        meta = svc.forecaster.meta[crop]
        met = meta["metrics"]
        test = met["split"]["test"]
        lines += [f"## {crop}", "",
                  f"Test origins {test[0]} → {test[1]} ({test[2]:,} forecasts), "
                  f"{meta['n_trees']} trees, 80% interval coverage "
                  f"{met['overall']['interval_coverage']:.0%}.", "",
                  "| Horizon / method | MAE | RMSE | MAPE | R² | Skill vs persistence |",
                  "| :-- | --: | --: | --: | --: | --: |"]
        for h, b in met["by_horizon"].items():
            for name in ("model", "moving_avg_7", "persistence"):
                label = {"model": f"**{h}-day — XGBoost**", "moving_avg_7": f"{h}-day — 7-day average",
                         "persistence": f"{h}-day — persistence"}[name]
                lines.append(_row(label, b[name], b["persistence"]))
        lines.append("")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return str(REPORT.relative_to(ROOT))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.perf_counter()
    svc = MandiService.start(retrain=True)
    print(f"Report written to {write_report(svc, time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
