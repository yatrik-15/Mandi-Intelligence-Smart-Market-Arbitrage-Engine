"""
recommend.py — Interactive command-line recommendation for a farmer.

    python scripts/recommend.py
    python scripts/recommend.py --crop Onion --qty 2000 --location Rajkot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_arbitrage.data_loader import TARGET_CROPS  # noqa: E402
from ml_arbitrage.distance_calculator import is_known_location  # noqa: E402
from ml_arbitrage.service import MandiService  # noqa: E402


def _ask(prompt: str, valid=lambda v: bool(v), cast=str):
    while True:
        raw = input(prompt).strip()
        try:
            value = cast(raw)
            if valid(value):
                return value
        except ValueError:
            pass
        print("  Invalid input, please try again.")


def display(rec: dict) -> None:
    if "error" in rec:
        print(f"\nError: {rec['error']}")
        return
    opt = rec["optimal_strategy"]
    bd = opt["cost_breakdown"]
    td = bd["transport_detail"]
    tag = "forecast" if opt["is_predicted_price"] else "today"
    print(f"\n=== {rec['recommendation']} ===")
    print(f"Mandi        : {opt['mandi']} ({opt['distance_km']:.0f} km)")
    print(f"Price        : Rs {opt['price_per_kg']:.2f}/kg ({tag})")
    print(f"Quantity sold: {bd['qty_remaining_kg']:,.0f} kg")
    print(f"Gross revenue: Rs {bd['gross_revenue']:>12,.0f}")
    print(f"Transport    : Rs {bd['transport_cost']:>12,.0f}  "
          f"({td['vehicle']}, {td['trips']} trip(s))")
    print(f"Storage      : Rs {bd['storage_cost']:>12,.0f}")
    print(f"Spoilage     : Rs {bd['perishability_cost']:>12,.0f}  (already in revenue)")
    print(f"NET PROFIT   : Rs {opt['net_profit']:>12,.0f}  "
          f"(80% range Rs {opt['net_profit_lower']:,.0f} – {opt['net_profit_upper']:,.0f})")
    print(f"\n{rec['justification']}")
    if rec["alternative_scenarios"]:
        print("\nOther mandis:")
        for s in rec["alternative_scenarios"]:
            when = "today" if s["days_to_wait"] == 0 else f"in {s['days_to_wait']} days"
            print(f"  {s['mandi_name']:<12} {s['distance_km']:>5.0f} km  {when:<11} "
                  f"Rs {s['net_profit']:>10,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Where and when should I sell?")
    parser.add_argument("--crop", choices=TARGET_CROPS)
    parser.add_argument("--qty", type=float, help="quantity in kg")
    parser.add_argument("--location", help="your town in Gujarat")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    crop = args.crop or _ask(f"Crop {TARGET_CROPS}: ", lambda v: v in TARGET_CROPS, str.title)
    qty = args.qty or _ask("Quantity (kg): ", lambda v: 0 < v <= 100_000, float)
    location = args.location or _ask("Your town: ", is_known_location)
    if not is_known_location(location):
        sys.exit(f"Unknown location: {location}")

    svc = MandiService.start()
    display(svc.recommend(crop, qty, location))


if __name__ == "__main__":
    main()
