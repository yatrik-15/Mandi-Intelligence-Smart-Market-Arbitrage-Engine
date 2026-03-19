"""
stage2_arbitrage.py — Stage 2: Deterministic Arbitrage Mathematics
===================================================================
Pure, side-effect-free calculation of Net Profit for each logistical
action scenario. No ML here — this is the economic core of the engine.

Net Profit Formula
------------------
    Net_Profit = (Price_future × Qty)
               − (Distance × FuelCost_per_km)
               − (WaitDays × PerishabilityFactor × Qty × Price_future)
               − (WaitDays × StorageCost_per_day)

Where:
    PerishabilityFactor captures the exponential value loss of the
    crop over time:  value_remaining = Qty × exp(−λ × t)
    Net degradation cost = Qty × Price_future × (1 − exp(−λ × t))
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArbitrageScenario:
    """Represents one logistical option and its economic outcome."""
    label:         str
    wait_days:     int
    net_profit:    float
    gross_revenue: float
    travel_cost:   float
    spoilage_cost: float
    storage_cost:  float


class ArbitrageMathEngine:
    """
    Calculates net profit for all logistical scenarios and returns
    a ranked list of ArbitrageScenario objects.

    The exponential decay model is more realistic than a linear one:
    perishable crops lose value fastest in the first few days post-harvest.
    """

    SCENARIOS = [
        {"label": "SELL_LOCAL_NOW",       "wait_days": 0},
        {"label": "TRAVEL_TO_FAR_MANDI",  "wait_days": 3},
        {"label": "STORE_FOR_T_DAYS",     "wait_days": 7},
    ]

    def net_profit(
        self,
        price_future:          float,
        qty:                   float,
        distance_km:           float,
        fuel_cost_per_km:      float,
        wait_days:             int,
        perishability_rate:    float,   # λ: fraction decaying per day
        storage_cost_per_day:  float,
    ) -> float:
        """
        Compute scalar net profit for a single scenario.

        Decay model: qty_remaining = qty × exp(−λ × t)
        Spoilage cost = (qty − qty_remaining) × price_future
        """
        qty_remaining  = qty * math.exp(-perishability_rate * wait_days)
        gross_revenue  = qty_remaining * price_future
        travel_cost    = distance_km * fuel_cost_per_km
        spoilage_cost  = (qty - qty_remaining) * price_future
        storage_cost   = wait_days * storage_cost_per_day

        return gross_revenue - travel_cost - spoilage_cost - storage_cost

    def evaluate_all_scenarios(
        self,
        price_t3:              float,
        price_t7:              float,
        price_now:             float,
        qty:                   float,
        distance_km:           float,
        fuel_cost_per_km:      float,
        perishability_rate:    float,
        storage_cost_per_day:  float,
    ) -> List[ArbitrageScenario]:
        """
        Evaluate all three canonical scenarios and return ranked list.

        Returns
        -------
        List[ArbitrageScenario], sorted descending by net_profit.
        """
        price_map = {0: price_now, 3: price_t3, 7: price_t7}
        results: List[ArbitrageScenario] = []

        for s in self.SCENARIOS:
            wait  = s["wait_days"]
            pfut  = price_map[wait]

            qty_remaining = qty * math.exp(-perishability_rate * wait)
            gross_revenue = qty_remaining * pfut
            travel_cost   = distance_km * fuel_cost_per_km
            spoilage_cost = (qty - qty_remaining) * pfut
            stor_cost     = wait * storage_cost_per_day
            net           = gross_revenue - travel_cost - spoilage_cost - stor_cost

            results.append(ArbitrageScenario(
                label         = s["label"],
                wait_days     = wait,
                net_profit    = round(net, 2),
                gross_revenue = round(gross_revenue, 2),
                travel_cost   = round(travel_cost, 2),
                spoilage_cost = round(spoilage_cost, 2),
                storage_cost  = round(stor_cost, 2),
            ))

        return sorted(results, key=lambda x: x.net_profit, reverse=True)
