"""
stage1a_prophet.py — Stage 1A: Macro Trend Extraction via Prophet
=================================================================
Responsibilities:
  1. Impute missing dates / price gaps via Prophet's built-in handling.
  2. Decompose historical price series into: trend + weekly + yearly components.
  3. Return an enriched DataFrame whose component columns are fed as
     known-future covariates into the TFT (Stage 1B).

WHY PROPHET HERE (not ARIMA):
  - Prophet natively handles irregular calendars, Indian public holidays,
    and multiple seasonality periods (Rabi/Kharif harvest cycles map to
    yearly seasonality; weekly Mandi days map to weekly seasonality).
  - It is robust to structural breaks (crop failures, price shocks) via
    automatic changepoint detection — ARIMA would require manual differencing
    and stationarity tests for each series, a maintenance nightmare at scale.
  - The decomposed components (trend, weekly, yearly) act as interpretable
    "prior knowledge" features that dramatically reduce the learning burden
    on the downstream TFT.
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

    Parameters
    ----------
    config : ProphetConfig
        Hyperparameters (changepoint flexibility, seasonality mode, etc.)
    country_holidays : str, optional
        ISO country code for automatic holiday injection (default: "IN" for India).
    """

    def __init__(self, config: ProphetConfig, country_holidays: str = "IN") -> None:
        self.config            = config
        self.country_holidays  = country_holidays
        self._model: Optional[Prophet] = None

    # ─────────────────────────────────────────────────────────────────────────

    def fit_transform(self, price_series: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Prophet on the price series and return a DataFrame with
        decomposed components aligned to the original dates.

        Parameters
        ----------
        price_series : pd.DataFrame
            Must contain columns ['ds' (datetime), 'y' (price)].

        Returns
        -------
        pd.DataFrame
            Original series + ['trend', 'weekly', 'yearly',
                                'yhat', 'yhat_lower', 'yhat_upper']
        """
        df = self._preprocess(price_series)

        logger.debug("Fitting Prophet on %d rows …", len(df))
        model = self._build_model()
        model.fit(df)
        self._model = model

        # In-sample forecast to extract decomposition
        forecast = model.predict(df[["ds"]])

        result = df.merge(
            forecast[["ds", "trend", "weekly", "yearly",
                       "yhat", "yhat_lower", "yhat_upper"]],
            on="ds",
            how="left",
        )

        # Fill any NaN components with forward-fill then zero
        for col in ["trend", "weekly", "yearly"]:
            result[col] = result[col].ffill().fillna(0.0)

        logger.info(
            "Prophet fit complete. "
            "Trend range: [%.2f, %.2f]  Weekly amplitude: %.4f",
            result["trend"].min(), result["trend"].max(),
            result["weekly"].abs().max(),
        )
        return result

    def predict_future(self, periods: int = 7) -> pd.DataFrame:
        """
        Extend the fitted model forward by `periods` days.
        Returns trend/seasonality components for the forecast window.
        """
        if self._model is None:
            raise RuntimeError("Must call fit_transform() before predict_future().")

        future = self._model.make_future_dataframe(periods=periods, freq="D")
        forecast = self._model.predict(future)
        return forecast[["ds", "trend", "weekly", "yearly",
                          "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_model(self) -> Prophet:
        model = Prophet(
            changepoint_prior_scale  = self.config.changepoint_prior_scale,
            seasonality_prior_scale  = self.config.seasonality_prior_scale,
            holidays_prior_scale     = self.config.holidays_prior_scale,
            seasonality_mode         = self.config.seasonality_mode,
            yearly_seasonality       = self.config.yearly_seasonality,
            weekly_seasonality       = self.config.weekly_seasonality,
            daily_seasonality        = self.config.daily_seasonality,
            interval_width           = self.config.interval_width,
            mcmc_samples             = self.config.mcmc_samples,
        )

        # Inject Indian agricultural seasonality (Rabi + Kharif cycles)
        model.add_seasonality(name="kharif", period=182.5, fourier_order=5)
        model.add_seasonality(name="rabi",   period=182.5, fourier_order=5, prior_scale=8.0)

        # Add Indian public/market holidays
        if self.country_holidays:
            try:
                model.add_country_holidays(country_name=self.country_holidays)
            except Exception as e:
                logger.warning("Could not load holidays for %s: %s", self.country_holidays, e)

        return model

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce dtypes, sort, fill exact duplicate dates, and cap outliers
        using a 3-sigma winsorisation before Prophet ingestion.
        """
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").drop_duplicates(subset=["ds"])

        # Fill missing dates with linear interpolation
        full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
        df = (
            df.set_index("ds")
              .reindex(full_range)
              .rename_axis("ds")
              .reset_index()
        )
        df["y"] = df["y"].interpolate(method="linear").fillna(method="bfill")

        # 3-sigma winsorisation (robust to flash price spikes)
        mu, sigma = df["y"].mean(), df["y"].std()
        df["y"] = df["y"].clip(mu - 3 * sigma, mu + 3 * sigma)

        return df
