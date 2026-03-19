"""
Stage 1A: Macro Trend Extraction via Prophet
==============================================
Improved version with Indian Agri-seasonality, Holidays, and Winsorization.
"""

from __future__ import annotations
import logging
from typing import Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from .config import ProphetConfig

logger = logging.getLogger(__name__)

class ProphetTrendExtractor:
    """
    Wraps Facebook Prophet for agricultural price trend decomposition.
    """

    def __init__(self, config: ProphetConfig, country_holidays: str = "IN") -> None:
        self.config            = config
        self.country_holidays  = country_holidays
        self._model: Optional[Prophet] = None

    def fit_transform(self, price_series: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Prophet and return decomposed components.
        price_series: columns ['ds', 'y']
        """
        df = self._preprocess(price_series)
        model = self._build_model()
        model.fit(df)
        self._model = model

        forecast = model.predict(df[["ds"]])
        
        # Merge with original (preprocessed) data
        result = df.merge(
            forecast[["ds", "trend", "weekly", "yearly", "yhat"]],
            on="ds",
            how="left",
        )

        for col in ["trend", "weekly", "yearly"]:
            result[col] = result[col].ffill().fillna(0.0)

        return result

    def _build_model(self) -> Prophet:
        model = Prophet(
            changepoint_prior_scale  = self.config.changepoint_prior_scale,
            seasonality_prior_scale  = self.config.seasonality_prior_scale,
            holidays_prior_scale     = self.config.holidays_prior_scale,
            seasonality_mode         = self.config.seasonality_mode,
            yearly_seasonality       = self.config.yearly_seasonality,
            weekly_seasonality       = self.config.weekly_seasonality,
            daily_seasonality        = self.config.daily_seasonality,
        )

        # Indian Agri-seasonality (Harvest cycles)
        model.add_seasonality(name="kharif", period=182.5, fourier_order=5)
        model.add_seasonality(name="rabi",   period=182.5, fourier_order=5, prior_scale=8.0)

        if self.country_holidays:
            try:
                model.add_country_holidays(country_name=self.country_holidays)
            except Exception as e:
                logger.warning("Could not load holidays: %s", e)

        return model

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").drop_duplicates(subset=["ds"])

        # Interpolate gaps
        full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
        df = df.set_index("ds").reindex(full_range).rename_axis("ds").reset_index()
        df["y"] = df["y"].interpolate(method="linear").bfill()

        # 3-sigma winsorisation for outliers
        mu, sigma = df["y"].mean(), df["y"].std()
        df["y"] = df["y"].clip(mu - 3 * sigma, mu + 3 * sigma)

        return df
