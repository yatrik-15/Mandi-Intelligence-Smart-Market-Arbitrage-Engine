"""
Stage 2: Deterministic Arbitrage Mathematics
==============================================
Improved version with Exponential Decay Model for perishability.
"""

import math
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ArbitrageScenario:
    label:         str
    wait_days:     int
    net_profit:    float

class ArbitrageMathEngine:
    """
    Economic core using exponential value loss.
    """

    def net_profit(
        self,
        price_future:          float,
        qty:                   float,
        distance_km:           float,
        fuel_cost_per_km:      float,
        wait_days:             int,
        perishability_rate:    float,   # λ (lambda)
        storage_cost_per_day:  float,
    ) -> float:
        """
        qty_remaining = qty * exp(-λ * t)
        """
        qty_remaining  = qty * math.exp(-perishability_rate * wait_days)
        gross_revenue  = qty_remaining * price_future
        travel_cost    = distance_km * fuel_cost_per_km
        spoilage_cost  = (qty - qty_remaining) * price_future
        storage_cost   = wait_days * storage_cost_per_day

        return gross_revenue - travel_cost - spoilage_cost - storage_cost

    def evaluate_scenarios(self, data: dict) -> List[ArbitrageScenario]:
        """
        Evaluate canonical scenarios.
        """
        scenarios = [
            ("SELL_LOCAL_NOW", 0, data['price_now']),
            ("TRAVEL_TO_FAR_MANDI", 3, data['price_t3']),
            ("STORE_FOR_T_DAYS", 7, data['price_t7']),
        ]
        
        results = []
        for label, wait, p_fut in scenarios:
            net = self.net_profit(
                price_future = p_fut,
                qty = data['qty'],
                distance_km = data['distance'] if wait == 3 else 0,
                fuel_cost_per_km = data['fuel_cost'],
                wait_days = wait,
                perishability_rate = data['perishability_rate'],
                storage_cost_per_day = data['storage_cost']
            )
            results.append(ArbitrageScenario(label, wait, round(net, 2)))
            
        return sorted(results, key=lambda x: x.net_profit, reverse=True)
