"""
Stage 1B: Multi-Horizon Price Forecasting via TFT
===================================================
Improved version with Group Normalization and Rolling Features.
"""

from __future__ import annotations
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from .config import TFTConfig

class TFTPriceForecaster:
    """
    Wraps pytorch-forecasting's TFT for price prediction.
    """

    def __init__(self, config: TFTConfig) -> None:
        self.config = config
        self._model: Optional[TemporalFusionTransformer] = None

    def _prepare_inference_df(self, tft_input: dict) -> pd.DataFrame:
        ts = tft_input["time_series"].copy()
        static = tft_input["static_covariates"]
        cats = tft_input["categoricals"]

        ts["time_idx"] = np.arange(len(ts))
        ts["mandi_id"] = static["mandi_id"]
        ts["lat"] = static["lat"]
        ts["lon"] = static["lon"]
        ts["crop_name"] = cats["crop_name"]
        ts["weather_condition"] = cats["weather"]

        # Engineered Rolling features
        ts["rolling_mean_7d"] = ts["y"].rolling(7, min_periods=1).mean()
        ts["rolling_mean_30d"] = ts["y"].rolling(30, min_periods=1).mean()

        ts["ds"] = pd.to_datetime(ts["ds"])
        ts["day_of_week"] = ts["ds"].dt.dayofweek.astype(float)
        ts["month"] = ts["ds"].dt.month.astype(float)

        ts = ts.rename(columns={
            "y": "price",
            "trend": "prophet_trend",
            "weekly": "prophet_weekly",
            "yearly": "prophet_yearly",
        })
        return ts

    def _build_dataset(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="price",
            group_ids=["mandi_id"],
            max_encoder_length=self.config.max_encoder_length,
            max_prediction_length=self.config.max_prediction_length,
            static_reals=self.config.static_reals,
            static_categoricals=self.config.static_categoricals,
            time_varying_known_reals=self.config.time_varying_known_reals,
            time_varying_known_categoricals=self.config.time_varying_known_categoricals,
            time_varying_unknown_reals=self.config.time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["mandi_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            allow_missing_timesteps=True,
        )

    def forecast(self, tft_input: dict) -> tuple[float, float]:
        """
        Simulated inference for the skeleton. In production, this would 
        load a checkpoint and run model.predict().
        """
        # (In production, load model here if not loaded)
        
        # Placeholder logic consistent with reference:
        # returns (price_t3, price_t7)
        latest_price = tft_input["time_series"]["y"].iloc[-1]
        price_t3 = latest_price * (1 + np.random.normal(0, 0.02))
        price_t7 = latest_price * (1 + np.random.normal(0, 0.05))
        
        return float(price_t3), float(price_t7)
