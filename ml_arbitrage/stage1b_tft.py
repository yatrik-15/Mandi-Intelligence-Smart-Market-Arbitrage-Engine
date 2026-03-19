"""
stage1b_tft.py — Stage 1B: Multi-Horizon Price Forecasting via TFT
===================================================================
Temporal Fusion Transformer (Lim et al., 2021) implemented via
`pytorch-forecasting` on top of PyTorch Lightning.

WHY TFT OUTPERFORMS XGBOOST FOR THIS TASK:
  - XGBoost is a tree-based tabular model; it cannot natively model
    temporal dependencies or attend to relevant historical windows.
    It treats each row as independent — catastrophic for sequential
    agricultural price data.
  - TFT's Variable Selection Networks (VSNs) learn, per timestep,
    WHICH inputs (prophet_trend, price, weather, rolling avg) are
    relevant — this is critical when some covariates (e.g., weather)
    only matter during harvest months.
  - The multi-head attention mechanism allows TFT to directly attend
    to analogous historical windows (e.g., same crop, same month,
    previous year) — encoding crop-cycle memory that XGBoost cannot.
  - Static covariates (GPS lat/lon) are injected via GRN encoders,
    letting the model learn geo-spatial price correlation implicitly.
  - TFT produces calibrated prediction intervals, enabling risk-aware
    logistics decisions rather than point estimates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader

from .config import TFTConfig

logger = logging.getLogger(__name__)

_CHECKPOINT_NAME = "tft_best.ckpt"


class TFTPriceForecaster:
    """
    Wraps pytorch-forecasting's TFT for t+3 and t+7 price prediction.

    The model:
      - Receives raw price time series + Prophet components as
        time-varying known reals.
      - Accepts GPS coords as static reals.
      - Produces quantile forecasts (0.1, 0.5, 0.9) — median is used
        as point estimate; quantiles drive confidence scoring.
    """

    def __init__(self, config: TFTConfig, model_dir: Path) -> None:
        self.config    = config
        self.model_dir = model_dir
        self._model:   Optional[TemporalFusionTransformer] = None
        self._dataset: Optional[TimeSeriesDataSet]         = None

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> None:
        """
        Train the TFT on a multi-series panel DataFrame.

        Parameters
        ----------
        train_df : pd.DataFrame
            Long-format panel. Expected columns:
              time_idx, mandi_id, price, prophet_trend, prophet_weekly,
              prophet_yearly, rolling_mean_7d, rolling_mean_30d,
              day_of_week, month, weather_condition, crop_name, lat, lon
        val_df : pd.DataFrame, optional
            Validation split. If None, last 20 % of train_df is used.
        """
        logger.info("Building TFT TimeSeriesDataSet …")

        if val_df is None:
            cutoff   = int(len(train_df) * 0.80)
            val_df   = train_df.iloc[cutoff:].copy()
            train_df = train_df.iloc[:cutoff].copy()

        self._dataset = self._build_dataset(train_df, is_train=True)
        val_dataset   = TimeSeriesDataSet.from_dataset(
            self._dataset, val_df, predict=True, stop_randomization=True
        )

        train_loader = self._dataset.to_dataloader(
            train=True, batch_size=self.config.batch_size, num_workers=0
        )
        val_loader = val_dataset.to_dataloader(
            train=False, batch_size=self.config.batch_size * 2, num_workers=0
        )

        model = TemporalFusionTransformer.from_dataset(
            self._dataset,
            learning_rate             = self.config.learning_rate,
            hidden_size               = self.config.hidden_size,
            attention_head_size       = self.config.attention_head_size,
            dropout                   = self.config.dropout,
            hidden_continuous_size    = self.config.hidden_size // 2,
            lstm_layers               = self.config.lstm_layers,
            output_size               = 7,   # 7 quantiles
            loss                      = QuantileLoss(),
            reduce_on_plateau_patience= self.config.reduce_on_plateau_patience,
            log_interval              = 10,
            log_val_interval          = 1,
        )

        logger.info("TFT parameters: %s", f"{model.hparams}")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.config.reduce_on_plateau_patience + 4),
            LearningRateMonitor(),
        ]

        trainer = pl.Trainer(
            max_epochs               = self.config.max_epochs,
            gradient_clip_val        = self.config.gradient_clip_val,
            callbacks                = callbacks,
            enable_progress_bar      = True,
            enable_model_summary     = True,
        )

        logger.info("Starting TFT training for max %d epochs …", self.config.max_epochs)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_path = self.model_dir / _CHECKPOINT_NAME
        trainer.save_checkpoint(str(best_path))
        self._model = model
        logger.info("TFT training complete. Checkpoint saved to: %s", best_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def forecast(self, tft_input: Dict) -> Tuple[float, float]:
        """
        Run inference for a single MandiRecord and return (price_t3, price_t7).

        Parameters
        ----------
        tft_input : dict
            Produced by MandiArbitrageEngine._prepare_tft_input().
            Keys: 'time_series', 'static_covariates', 'categoricals'

        Returns
        -------
        (price_t3, price_t7) : Tuple[float, float]
            Median quantile forecasts at t+3 and t+7.
        """
        if self._model is None:
            self._try_load_checkpoint()

        inference_df = self._prepare_inference_df(tft_input)
        pred_dataset = self._build_dataset(inference_df, is_train=False)
        pred_loader  = pred_dataset.to_dataloader(
            train=False, batch_size=1, num_workers=0
        )

        self._model.eval()
        with torch.no_grad():
            preds, _ = self._model.predict(pred_loader, return_index=True, mode="quantiles")

        # preds shape: (n_series, prediction_length, n_quantiles)
        # Quantile index 3 = median (0.5) in default QuantileLoss
        median_preds = preds[0, :, 3].numpy()  # shape: (max_prediction_length,)

        price_t3 = float(median_preds[min(2, len(median_preds) - 1)])
        price_t7 = float(median_preds[min(6, len(median_preds) - 1)])

        logger.debug("TFT forecast → t+3: %.2f  t+7: %.2f", price_t3, price_t7)
        return price_t3, price_t7

    def load_checkpoint(self) -> None:
        self._try_load_checkpoint()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_dataset(self, df: pd.DataFrame, is_train: bool) -> TimeSeriesDataSet:
        cfg = self.config
        return TimeSeriesDataSet(
            df,
            time_idx                         = "time_idx",
            target                           = "price",
            group_ids                        = ["mandi_id"],
            max_encoder_length               = cfg.max_encoder_length,
            max_prediction_length            = cfg.max_prediction_length,
            static_reals                     = cfg.static_reals,
            static_categoricals              = cfg.static_categoricals,
            time_varying_known_reals         = cfg.time_varying_known_reals,
            time_varying_known_categoricals  = cfg.time_varying_known_categoricals,
            time_varying_unknown_reals       = cfg.time_varying_unknown_reals,
            target_normalizer                = GroupNormalizer(
                groups=["mandi_id"], transformation="softplus"
            ),
            add_relative_time_idx            = True,
            add_target_scales                = True,
            add_encoder_length               = True,
            allow_missing_timesteps          = True,
        )

    def _prepare_inference_df(self, tft_input: Dict) -> pd.DataFrame:
        """
        Converts the dict produced by _prepare_tft_input into a
        TimeSeriesDataSet-compatible long-format DataFrame.
        """
        ts      = tft_input["time_series"].copy()
        static  = tft_input["static_covariates"]
        cats    = tft_input["categoricals"]

        ts["time_idx"]         = np.arange(len(ts))
        ts["mandi_id"]         = static["mandi_id"]
        ts["lat"]              = static["lat"]
        ts["lon"]              = static["lon"]
        ts["crop_name"]        = cats["crop_name"]
        ts["weather_condition"]= cats["weather"]

        # Rolling features
        ts["rolling_mean_7d"]  = ts["y"].rolling(7,  min_periods=1).mean()
        ts["rolling_mean_30d"] = ts["y"].rolling(30, min_periods=1).mean()

        # Calendar features
        ts["ds"]               = pd.to_datetime(ts["ds"])
        ts["day_of_week"]      = ts["ds"].dt.dayofweek.astype(str)
        ts["month"]            = ts["ds"].dt.month.astype(str)

        # Align column names to TFT config
        ts = ts.rename(columns={
            "y":            "price",
            "trend":        "prophet_trend",
            "weekly":       "prophet_weekly",
            "yearly":       "prophet_yearly",
        })

        return ts

    def _try_load_checkpoint(self) -> None:
        ckpt = self.model_dir / _CHECKPOINT_NAME
        if ckpt.exists():
            logger.info("Loading TFT checkpoint from %s", ckpt)
            self._model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
        else:
            raise FileNotFoundError(
                f"TFT checkpoint not found at {ckpt}. "
                "Call TFTPriceForecaster.fit() first."
            )
