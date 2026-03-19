"""
Mandi Arbitrage Engine — Core Pipeline Orchestrator
=====================================================
Three-Stage Ensemble: Prophet → TFT → CatBoost
Author  : Senior ML Engineer
Version : 2.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .stage1a_prophet   import ProphetTrendExtractor
from .stage1b_tft       import TFTPriceForecaster
from .stage2_arbitrage  import ArbitrageMathEngine
from .stage3_catboost   import CatBoostDecisionClassifier
from .evaluation        import ModelEvaluator
from .config            import EngineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MandiArbitrageEngine")


# ──────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MandiRecord:
    """Atomic unit representing one Mandi's full feature set."""
    mandi_id:               str
    crop_name:              str
    lat:                    float
    lon:                    float
    price_series:           pd.DataFrame   # cols: [ds, y]  (date, price)
    weather_condition:      str
    distance_km:            float
    fuel_cost_per_km:       float
    perishability_decay_rate: float        # fraction per day  e.g. 0.05 = 5 %/day
    storage_cost_per_day:   float
    qty_available:          float          # quintals

    def validate(self) -> None:
        required_cols = {"ds", "y"}
        missing = required_cols - set(self.price_series.columns)
        if missing:
            raise ValueError(f"price_series missing columns: {missing}")
        if len(self.price_series) < 60:
            raise ValueError("Need ≥ 60 days of price history for reliable forecasts.")
        if not (0.0 <= self.perishability_decay_rate <= 1.0):
            raise ValueError("perishability_decay_rate must be in [0, 1].")


@dataclass
class ArbitrageDecision:
    """Final enriched output per Mandi."""
    mandi_id:       str
    crop_name:      str
    action:         str                    # SELL_LOCAL_NOW | TRAVEL_TO_FAR_MANDI | STORE_FOR_T_DAYS
    action_proba:   Dict[str, float]       # class → probability
    price_t3:       float
    price_t7:       float
    net_profit_t3:  float
    net_profit_t7:  float
    confidence:     float
    reasoning:      str


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class MandiArbitrageEngine:
    """
    Orchestrates the full three-stage pipeline.

    Stage 1A — Prophet  : Macro trend decomposition + gap imputation
    Stage 1B — TFT      : Multi-horizon price forecast  (t+3, t+7)
    Stage 2  — Math     : Net Profit computation per logistics scenario
    Stage 3  — CatBoost : Optimal action classification
    """

    def __init__(self, config: Optional[EngineConfig] = None, model_dir: str = "models/") -> None:
        self.config   = config or EngineConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.prophet_extractor   = ProphetTrendExtractor(self.config.prophet)
        self.tft_forecaster      = TFTPriceForecaster(self.config.tft, self.model_dir)
        self.arbitrage_engine    = ArbitrageMathEngine()
        self.catboost_classifier = CatBoostDecisionClassifier(self.config.catboost, self.model_dir)
        self.evaluator           = ModelEvaluator()

        logger.info("MandiArbitrageEngine initialised. Config: %s", self.config)

    # ──────────────────────────── Public API ──────────────────────────────────

    def train(self, records: List[MandiRecord], labels: pd.Series) -> None:
        """
        Full supervised training pass.

        Parameters
        ----------
        records : list of MandiRecord
            Historical mandi records WITH known outcomes.
        labels  : pd.Series
            Ground-truth action labels aligned to `records`.
        """
        logger.info("═══ TRAINING PIPELINE ═══")

        feature_rows: List[Dict] = []

        for rec in records:
            rec.validate()
            row = self._build_feature_row(rec)
            feature_rows.append(row)

        X_train = pd.DataFrame(feature_rows)
        logger.info("Training CatBoost on %d samples …", len(X_train))
        self.catboost_classifier.train(X_train, labels)
        logger.info("Training complete.")

    def predict(self, record: MandiRecord) -> ArbitrageDecision:
        """
        Run the full pipeline for a single MandiRecord and return
        a fully enriched ArbitrageDecision.
        """
        record.validate()
        logger.info("Processing mandi_id=%s crop=%s", record.mandi_id, record.crop_name)

        row = self._build_feature_row(record)
        X   = pd.DataFrame([row])

        action, proba = self.catboost_classifier.predict(X)

        return ArbitrageDecision(
            mandi_id      = record.mandi_id,
            crop_name     = record.crop_name,
            action        = action[0],
            action_proba  = proba[0],
            price_t3      = row["price_t3"],
            price_t7      = row["price_t7"],
            net_profit_t3 = row["net_profit_t3"],
            net_profit_t7 = row["net_profit_t7"],
            confidence    = max(proba[0].values()),
            reasoning     = self._generate_reasoning(row, action[0], proba[0]),
        )

    def predict_batch(self, records: List[MandiRecord]) -> List[ArbitrageDecision]:
        """Vectorised batch prediction with shared TFT inference."""
        [r.validate() for r in records]
        rows = [self._build_feature_row(r) for r in records]
        X    = pd.DataFrame(rows)
        actions, probas = self.catboost_classifier.predict(X)
        decisions = []
        for i, rec in enumerate(records):
            decisions.append(ArbitrageDecision(
                mandi_id      = rec.mandi_id,
                crop_name     = rec.crop_name,
                action        = actions[i],
                action_proba  = probas[i],
                price_t3      = rows[i]["price_t3"],
                price_t7      = rows[i]["price_t7"],
                net_profit_t3 = rows[i]["net_profit_t3"],
                net_profit_t7 = rows[i]["net_profit_t7"],
                confidence    = max(probas[i].values()),
                reasoning     = self._generate_reasoning(rows[i], actions[i], probas[i]),
            ))
        return decisions

    def evaluate(
        self,
        records:      List[MandiRecord],
        true_labels:  pd.Series,
        true_prices:  Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Comprehensive evaluation block.

        Returns RMSE for TFT price forecasts (if true_prices supplied)
        and Precision / Recall / F1 for CatBoost classifications.
        """
        rows         = [self._build_feature_row(r) for r in records]
        X            = pd.DataFrame(rows)
        actions, _   = self.catboost_classifier.predict(X)
        clf_metrics  = self.evaluator.classification_report(true_labels.tolist(), actions)

        results: Dict = {"classification": clf_metrics}

        if true_prices:
            pred_t3 = [r["price_t3"] for r in rows]
            pred_t7 = [r["price_t7"] for r in rows]
            true_t3 = [true_prices.get(f"{rec.mandi_id}_t3", np.nan) for rec in records]
            true_t7 = [true_prices.get(f"{rec.mandi_id}_t7", np.nan) for rec in records]
            results["tft_rmse_t3"] = self.evaluator.rmse(true_t3, pred_t3)
            results["tft_rmse_t7"] = self.evaluator.rmse(true_t7, pred_t7)

        self.evaluator.print_report(results)
        return results

    # ──────────────────────── Internal Helpers ────────────────────────────────

    def _build_feature_row(self, rec: MandiRecord) -> Dict:
        """
        Executes Stage 1A → 1B → 2 and returns a flat feature dict
        ready for CatBoost ingestion.
        """
        # ── Stage 1A: Prophet macro trend ────────────────────────────────────
        trend_df = self.prophet_extractor.fit_transform(rec.price_series)

        # ── Stage 1B: TFT price forecast  ────────────────────────────────────
        tft_input = self._prepare_tft_input(rec, trend_df)
        price_t3, price_t7 = self.tft_forecaster.forecast(tft_input)

        # ── Stage 2: Arbitrage math  ──────────────────────────────────────────
        net_profit_t3 = self.arbitrage_engine.net_profit(
            price_future         = price_t3,
            qty                  = rec.qty_available,
            distance_km          = rec.distance_km,
            fuel_cost_per_km     = rec.fuel_cost_per_km,
            wait_days            = 3,
            perishability_rate   = rec.perishability_decay_rate,
            storage_cost_per_day = rec.storage_cost_per_day,
        )
        net_profit_t7 = self.arbitrage_engine.net_profit(
            price_future         = price_t7,
            qty                  = rec.qty_available,
            distance_km          = rec.distance_km,
            fuel_cost_per_km     = rec.fuel_cost_per_km,
            wait_days            = 7,
            perishability_rate   = rec.perishability_decay_rate,
            storage_cost_per_day = rec.storage_cost_per_day,
        )

        # ── Flat feature row ─────────────────────────────────────────────────
        return {
            "mandi_id":                  rec.mandi_id,
            "crop_name":                 rec.crop_name,
            "lat":                       rec.lat,
            "lon":                       rec.lon,
            "weather_condition":         rec.weather_condition,
            "distance_km":               rec.distance_km,
            "fuel_cost_per_km":          rec.fuel_cost_per_km,
            "perishability_decay_rate":  rec.perishability_decay_rate,
            "storage_cost_per_day":      rec.storage_cost_per_day,
            "qty_available":             rec.qty_available,
            # Stage 1A outputs
            "prophet_trend_last":        trend_df["trend"].iloc[-1],
            "prophet_weekly_last":       trend_df["weekly"].iloc[-1],
            "prophet_yearly_last":       trend_df["yearly"].iloc[-1],
            # Stage 1B outputs
            "price_t3":                  price_t3,
            "price_t7":                  price_t7,
            # Stage 2 outputs
            "net_profit_t3":             net_profit_t3,
            "net_profit_t7":             net_profit_t7,
            # Derived signals
            "price_momentum_3d":         price_t3 - rec.price_series["y"].iloc[-1],
            "price_momentum_7d":         price_t7 - rec.price_series["y"].iloc[-1],
            "profit_ratio_t7_t3":        (net_profit_t7 / (net_profit_t3 + 1e-9)),
        }

    @staticmethod
    def _prepare_tft_input(rec: MandiRecord, trend_df: pd.DataFrame) -> Dict:
        """Merges static covariates, time-series, and Prophet outputs for TFT."""
        merged = rec.price_series.copy()
        merged = merged.merge(
            trend_df[["ds", "trend", "weekly", "yearly"]],
            on="ds", how="left"
        )
        return {
            "time_series":       merged,
            "static_covariates": {"lat": rec.lat, "lon": rec.lon, "mandi_id": rec.mandi_id},
            "categoricals":      {"crop_name": rec.crop_name, "weather": rec.weather_condition},
        }

    @staticmethod
    def _generate_reasoning(row: Dict, action: str, proba: Dict[str, float]) -> str:
        lines = [
            f"Action: {action}  (confidence {max(proba.values()):.1%})",
            f"Price forecast → t+3: ₹{row['price_t3']:.2f}  t+7: ₹{row['price_t7']:.2f}",
            f"Net profit     → t+3: ₹{row['net_profit_t3']:.2f}  t+7: ₹{row['net_profit_t7']:.2f}",
            f"Price momentum → 3d: {row['price_momentum_3d']:+.2f}  7d: {row['price_momentum_7d']:+.2f}",
            f"Prophet trend (last): {row['prophet_trend_last']:.2f}",
        ]
        return " | ".join(lines)
