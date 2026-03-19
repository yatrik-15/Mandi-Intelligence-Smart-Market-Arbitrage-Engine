"""
Mandi Arbitrage Engine — Core Pipeline Orchestrator
=====================================================
Refined version with Data Contracts and Automated Reasoning.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .prophet_model import ProphetTrendExtractor
from .tft_model import TFTPriceForecaster
from .arbitrage_math import ArbitrageMathEngine
from .decision_classifier import CatBoostDecisionClassifier
from .config import EngineConfig

logger = logging.getLogger(__name__)

@dataclass
class MandiRecord:
    mandi_id: str
    crop_name: str
    lat: float
    lon: float
    price_series: pd.DataFrame
    weather_condition: str
    distance_km: float
    fuel_cost_per_km: float
    perishability_decay_rate: float
    storage_cost_per_day: float
    qty_available: float

@dataclass
class ArbitrageDecision:
    mandi_id: str
    crop_name: str
    action: str
    action_proba: Dict[str, float]
    price_t3: float
    price_t7: float
    net_profit_t3: float
    net_profit_t7: float
    confidence: float
    reasoning: str

class MandiArbitrageEngine:
    """
    Refined Orchestrator.
    """

    def __init__(self, config: Optional[EngineConfig] = None, model_dir: str = "models/") -> None:
        self.config = config or EngineConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.prophet_extractor = ProphetTrendExtractor(self.config.prophet)
        self.tft_forecaster = TFTPriceForecaster(self.config.tft)
        self.arbitrage_engine = ArbitrageMathEngine()
        self.catboost_classifier = CatBoostDecisionClassifier(self.config.catboost, self.model_dir)

    def predict(self, record: MandiRecord) -> ArbitrageDecision:
        row = self._build_feature_row(record)
        X = pd.DataFrame([row])

        actions, probas = self.catboost_classifier.predict(X)
        action = actions[0]
        proba = probas[0]

        return ArbitrageDecision(
            mandi_id=record.mandi_id,
            crop_name=record.crop_name,
            action=action,
            action_proba=proba,
            price_t3=row["price_t3"],
            price_t7=row["price_t7"],
            net_profit_t3=row["net_profit_t3"],
            net_profit_t7=row["net_profit_t7"],
            confidence=max(proba.values()),
            reasoning=self._generate_reasoning(row, action, proba)
        )

    def _build_feature_row(self, rec: MandiRecord) -> Dict:
        # Stage 1A: Prophet macro trend
        trend_df = self.prophet_extractor.fit_transform(rec.price_series.rename(columns={'Date': 'ds', 'Price': 'y'}))
        
        # Stage 1B: TFT price forecast
        tft_input = {
            "time_series": trend_df,
            "static_covariates": {"lat": rec.lat, "lon": rec.lon, "mandi_id": rec.mandi_id},
            "categoricals": {"crop_name": rec.crop_name, "weather": rec.weather_condition},
        }
        price_t3, price_t7 = self.tft_forecaster.forecast(tft_input)

        # Stage 2: Arbitrage math
        # We also need the scalar profits for features
        net_t3 = self.arbitrage_engine.net_profit(price_t3, rec.qty_available, rec.distance_km, rec.fuel_cost_per_km, 3, rec.perishability_decay_rate, rec.storage_cost_per_day)
        net_t7 = self.arbitrage_engine.net_profit(price_t7, rec.qty_available, 0, rec.fuel_cost_per_km, 7, rec.perishability_decay_rate, rec.storage_cost_per_day)

        return {
            "mandi_id": rec.mandi_id,
            "crop_name": rec.crop_name,
            "lat": rec.lat,
            "lon": rec.lon,
            "weather_condition": rec.weather_condition,
            "distance_km": rec.distance_km,
            "fuel_cost_per_km": rec.fuel_cost_per_km,
            "perishability_decay_rate": rec.perishability_decay_rate,
            "storage_cost_per_day": rec.storage_cost_per_day,
            "qty_available": rec.qty_available,
            "prophet_trend_last": trend_df["trend"].iloc[-1],
            "prophet_weekly_last": trend_df["weekly"].iloc[-1],
            "prophet_yearly_last": trend_df["yearly"].iloc[-1],
            "price_t3": price_t3,
            "price_t7": price_t7,
            "net_profit_t3": net_t3,
            "net_profit_t7": net_t7,
            "price_momentum_3d": price_t3 - rec.price_series["Price"].iloc[-1],
            "price_momentum_7d": price_t7 - rec.price_series["Price"].iloc[-1],
            "profit_ratio_t7_t3": (net_t7 / (net_t3 + 1e-9)),
        }

    def _generate_reasoning(self, row: Dict, action: str, proba: Dict[str, float]) -> str:
        return f"Action: {action} ({max(proba.values()):.1%}) | Price forecast t+7: ₹{row['price_t7']:.2f} | Net profit t+7: ₹{row['net_profit_t7']:.2f}"
