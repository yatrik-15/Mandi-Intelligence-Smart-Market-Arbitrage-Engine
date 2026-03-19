"""
config.py — Centralised hyperparameter management for all three model stages.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class ProphetConfig:
    """Hyperparameters for Prophet macro-trend extraction (Stage 1A)."""
    changepoint_prior_scale:   float = 0.05
    seasonality_prior_scale:   float = 10.0
    holidays_prior_scale:      float = 10.0
    seasonality_mode:          str   = "multiplicative"
    yearly_seasonality:        bool  = True
    weekly_seasonality:        bool  = True
    daily_seasonality:         bool  = False
    interval_width:            float = 0.80
    mcmc_samples:              int   = 0


@dataclass
class TFTConfig:
    """Hyperparameters for the Temporal Fusion Transformer (Stage 1B)."""
    max_encoder_length:        int   = 60
    max_prediction_length:     int   = 7
    hidden_size:               int   = 64
    lstm_layers:               int   = 2
    dropout:                   float = 0.1
    attention_head_size:       int   = 4
    learning_rate:             float = 1e-3
    max_epochs:                int   = 50
    batch_size:                int   = 64
    gradient_clip_val:         float = 0.1
    reduce_on_plateau_patience:int   = 4
    
    time_varying_known_reals:   List[str] = field(default_factory=lambda: [
        "price", "prophet_trend", "prophet_weekly", "prophet_yearly",
        "rolling_mean_7d", "rolling_mean_30d", "day_of_week", "month",
    ])
    time_varying_unknown_reals: List[str] = field(default_factory=lambda: ["price"])
    static_reals:               List[str] = field(default_factory=lambda: ["lat", "lon"])
    static_categoricals:        List[str] = field(default_factory=lambda: ["mandi_id"])
    time_varying_known_categoricals: List[str] = field(default_factory=lambda: [
        "weather_condition", "crop_name",
    ])


@dataclass
class CatBoostConfig:
    """Hyperparameters for CatBoost Decision Classifier (Stage 3)."""
    iterations:        int   = 500
    learning_rate:     float = 0.05
    depth:             int   = 6
    l2_leaf_reg:       float = 3.0
    random_seed:       int   = 42
    eval_metric:       str   = "MultiClass"
    loss_function:     str   = "MultiClass"
    verbose:           int   = 50
    early_stopping_rounds: int = 30
    cat_features:      List[str] = field(default_factory=lambda: [
        "mandi_id", "crop_name", "weather_condition",
    ])
    feature_cols:      List[str] = field(default_factory=lambda: [
        "mandi_id", "crop_name", "lat", "lon", "weather_condition",
        "distance_km", "fuel_cost_per_km", "perishability_decay_rate",
        "storage_cost_per_day", "qty_available",
        "prophet_trend_last", "prophet_weekly_last", "prophet_yearly_last",
        "price_t3", "price_t7", "net_profit_t3", "net_profit_t7",
        "price_momentum_3d", "price_momentum_7d", "profit_ratio_t7_t3",
    ])
    class_names:       List[str] = field(default_factory=lambda: [
        "SELL_LOCAL_NOW", "TRAVEL_TO_FAR_MANDI", "STORE_FOR_T_DAYS",
    ])


@dataclass
class EngineConfig:
    """Top-level configuration container for the full Mandi Arbitrage Engine."""
    prophet:  ProphetConfig  = field(default_factory=ProphetConfig)
    tft:      TFTConfig      = field(default_factory=TFTConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    forecast_horizons: List[int] = field(default_factory=lambda: [3, 7])
