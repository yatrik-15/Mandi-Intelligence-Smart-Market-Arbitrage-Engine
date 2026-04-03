"""
Train Multi-Stage Pipeline: Prophet -> TFT -> CatBoost
========================================================
Full three-stage ensemble training for the Mandi Arbitrage Engine.

Stage 1A — Prophet  : Decompose each (mandi, crop) price series into
                      trend + weekly + yearly + Rabi/Kharif seasonality
Stage 1B — TFT      : Temporal Fusion Transformer for multi-horizon
                      price forecasting (t+3, t+7) using Prophet outputs
Stage 2  — Math     : Deterministic arbitrage profit calculation
Stage 3  — CatBoost : Classify optimal action (SELL_NOW/TRAVEL/STORE)

Designed for: NVIDIA RTX 4050 6GB + Python 3.13 + CUDA 12.4
"""

import sys
import os
import json
import time
import math
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATASET_OLD    = "dataset/commodity_price.csv"
DATASET_NEW    = "dataset/new_dataset.csv"
MODELS_DIR     = Path("ml_arbitrage/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# State filter & target crops (from data_loader)
STATE_FILTER  = "Gujarat"
TARGET_CROPS  = ["Onion", "Tomato", "Potato", "Wheat", "Cotton"]

# Prophet config
PROPHET_CFG = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
}

# TFT config
TFT_CFG = {
    "max_encoder_length": 60,
    "max_prediction_length": 7,
    "hidden_size": 32,         # Compact for 6GB VRAM
    "lstm_layers": 2,
    "dropout": 0.1,
    "attention_head_size": 2,
    "learning_rate": 1e-3,
    "max_epochs": 30,
    "batch_size": 64,
    "gradient_clip_val": 0.1,
}

# CatBoost config
CATBOOST_CFG = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": 50,
    "early_stopping_rounds": 30,
}

# Perishability rates (daily decay fraction)
PERISHABILITY = {
    "Cotton": 0.005, "Wheat": 0.008, "Onion": 0.030,
    "Tomato": 0.055, "Potato": 0.020,
}


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_step(step: int, msg: str) -> None:
    print(f"  [{step}] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_and_process_data() -> pd.DataFrame:
    """Load, combine, and process old + new datasets through the data pipeline."""
    from ml_arbitrage.data_loader import MandiDataLoader

    print_step(1, "Loading datasets...")

    loader = MandiDataLoader()

    # Try loading combined dataset first
    combined_path = "dataset/combined_dataset.csv"
    if os.path.exists(combined_path):
        print(f"      Using existing combined dataset: {combined_path}")
        loader.load_data(combined_path)
    else:
        # Combine old + new
        dfs = []
        for path in [DATASET_OLD, DATASET_NEW]:
            if os.path.exists(path):
                df = pd.read_csv(path, encoding="utf-8", low_memory=False)
                df.columns = (df.columns.str.strip()
                              .str.replace("_x0020_", "_", regex=False)
                              .str.replace(r"\s+", "_", regex=True))
                dfs.append(df)
                print(f"      Loaded {path}: {len(df):,} records")

        combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
        combined.to_csv(combined_path, index=False)
        print(f"      Combined: {len(combined):,} -> {combined_path}")
        loader.load_data(combined_path)

    # Process through pipeline
    df = loader.filter_and_process(days=None, state_filter=STATE_FILTER)
    print(f"      Processed: {len(df):,} records")
    print(f"      Mandis: {sorted(df['Mandi_Name'].unique().tolist())}")
    print(f"      Crops: {sorted(df['Crop'].unique().tolist())}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: PROPHET TREND DECOMPOSITION (Stage 1A)
# ──────────────────────────────────────────────────────────────────────────────

def train_prophet_all(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Prophet per (mandi, crop) and return enriched DataFrame."""
    from prophet import Prophet

    print_step(2, "Training Prophet models (trend decomposition)...")

    groups = df.groupby(["Mandi_Name", "Crop"])
    enriched_frames = []
    prophet_models = {}
    total = len(groups)
    done = 0

    for (mandi, crop), group in groups:
        done += 1
        if len(group) < 60:
            print(f"      SKIP {mandi}-{crop}: only {len(group)} records (need 60+)")
            continue

        # Prepare Prophet format
        series = group[["Date", "Price_per_kg"]].copy()
        series.columns = ["ds", "y"]
        series["ds"] = pd.to_datetime(series["ds"])
        series = series.sort_values("ds").drop_duplicates(subset=["ds"])

        # Fill missing dates with interpolation
        full_range = pd.date_range(series["ds"].min(), series["ds"].max(), freq="D")
        series = (series.set_index("ds")
                  .reindex(full_range)
                  .rename_axis("ds")
                  .reset_index())
        series["y"] = series["y"].interpolate(method="linear").bfill()

        # 3-sigma winsorisation
        mu, sigma = series["y"].mean(), series["y"].std()
        if sigma > 0:
            series["y"] = series["y"].clip(mu - 3 * sigma, mu + 3 * sigma)

        # Build and fit Prophet
        model = Prophet(
            **PROPHET_CFG,
            interval_width=0.80,
        )
        model.add_seasonality(name="kharif", period=182.5, fourier_order=5)
        model.add_seasonality(name="rabi", period=182.5, fourier_order=5, prior_scale=8.0)

        try:
            model.add_country_holidays(country_name="IN")
        except Exception:
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(series)

        # In-sample decomposition
        forecast = model.predict(series[["ds"]])
        decomposed = series.merge(
            forecast[["ds", "trend", "weekly", "yearly", "yhat", "yhat_lower", "yhat_upper"]],
            on="ds", how="left"
        )
        for col in ["trend", "weekly", "yearly"]:
            decomposed[col] = decomposed[col].ffill().fillna(0.0)

        # Merge back with original data
        merged = group.copy()
        merged["Date"] = pd.to_datetime(merged["Date"])
        merged = merged.merge(
            decomposed[["ds", "trend", "weekly", "yearly", "yhat", "yhat_lower", "yhat_upper"]],
            left_on="Date", right_on="ds", how="left"
        )
        merged.drop(columns=["ds"], inplace=True, errors="ignore")

        # Fill any NaN from merge
        for col in ["trend", "weekly", "yearly", "yhat", "yhat_lower", "yhat_upper"]:
            if col in merged.columns:
                merged[col] = merged[col].ffill().bfill().fillna(0.0)

        enriched_frames.append(merged)
        prophet_models[(mandi, crop)] = model

        print(f"      [{done}/{total}] {mandi}-{crop}: "
              f"trend [{decomposed['trend'].min():.2f}, {decomposed['trend'].max():.2f}] "
              f"weekly amp {decomposed['weekly'].abs().max():.4f}")

    # Save Prophet metadata
    prophet_meta = {}
    for (mandi, crop), model in prophet_models.items():
        prophet_meta[f"{mandi}_{crop}"] = {
            "mandi": mandi, "crop": crop,
            "trained_at": str(datetime.now()),
            "trend_range": [float(enriched_frames[-1]["trend"].min()),
                           float(enriched_frames[-1]["trend"].max())],
        }
    (MODELS_DIR / "prophet_models_meta.json").write_text(json.dumps(prophet_meta, indent=2))

    df_enriched = pd.concat(enriched_frames, ignore_index=True)
    print(f"      Prophet complete: {len(df_enriched):,} enriched records, "
          f"{len(prophet_models)} models")

    return df_enriched, prophet_models


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: TFT FORECASTING (Stage 1B)
# ──────────────────────────────────────────────────────────────────────────────

def train_tft(df_enriched: pd.DataFrame) -> Dict:
    """Train a Temporal Fusion Transformer for multi-horizon price forecasting."""
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

    print_step(3, "Training TFT (Temporal Fusion Transformer)...")

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"      GPU detected: {device_name}")
        print(f"      CUDA version: {torch.version.cuda}")
    else:
        print("      No GPU detected, training on CPU (will be slower)")

    # Prepare panel data
    df = df_enriched.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Mandi_Name", "Crop", "Date"])

    # Create time index (integer per group)
    df["time_idx"] = df.groupby(["Mandi_Name", "Crop"]).cumcount()

    # Create a combined group ID
    df["group_id"] = df["Mandi_Name"] + "_" + df["Crop"]

    # Rolling features
    for grp_cols in [["Mandi_Name", "Crop"]]:
        grouped = df.groupby(grp_cols)["Price_per_kg"]
        df["rolling_mean_7d"] = grouped.transform(lambda x: x.rolling(7, min_periods=1).mean())
        df["rolling_mean_30d"] = grouped.transform(lambda x: x.rolling(30, min_periods=1).mean())

    # Calendar features as categories
    df["day_of_week"] = df["Date"].dt.dayofweek.astype(str)
    df["month_cat"] = df["Date"].dt.month.astype(str)

    # Ensure numeric columns are float
    float_cols = ["Price_per_kg", "trend", "weekly", "yearly",
                  "rolling_mean_7d", "rolling_mean_30d", "Distance_km"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Filter groups with enough data for encoder + decoder
    min_len = TFT_CFG["max_encoder_length"] + TFT_CFG["max_prediction_length"] + 10
    group_sizes = df.groupby("group_id").size()
    valid_groups = group_sizes[group_sizes >= min_len].index
    df = df[df["group_id"].isin(valid_groups)].reset_index(drop=True)

    if len(df) == 0:
        print("      ERROR: No groups with enough data for TFT training!")
        return {"error": "insufficient_data"}

    print(f"      Panel data: {len(df):,} rows, {len(valid_groups)} groups")
    print(f"      Time range: {df['time_idx'].min()} -> {df['time_idx'].max()}")

    # Train/val split by time (last 20%)
    max_time = df["time_idx"].max()
    training_cutoff = int(max_time * 0.80)

    # Build time-varying known reals (features we know in the future)
    time_varying_known_reals = ["time_idx"]
    # Add Prophet components as known reals (they can be forecast)
    for col in ["trend", "weekly", "yearly"]:
        if col in df.columns:
            time_varying_known_reals.append(col)

    # Distance is known
    if "Distance_km" in df.columns:
        time_varying_known_reals.append("Distance_km")

    # Rolling means as unknown reals (we don't know future values)
    time_varying_unknown_reals = ["Price_per_kg"]
    for col in ["rolling_mean_7d", "rolling_mean_30d"]:
        if col in df.columns:
            time_varying_unknown_reals.append(col)

    print(f"      Known reals: {time_varying_known_reals}")
    print(f"      Unknown reals: {time_varying_unknown_reals}")

    # Create dataset
    try:
        training_ds = TimeSeriesDataSet(
            df[df.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="Price_per_kg",
            group_ids=["group_id"],
            max_encoder_length=TFT_CFG["max_encoder_length"],
            max_prediction_length=TFT_CFG["max_prediction_length"],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            time_varying_known_categoricals=["day_of_week", "month_cat"],
            target_normalizer=GroupNormalizer(
                groups=["group_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
    except Exception as e:
        print(f"      ERROR creating TimeSeriesDataSet: {e}")
        print("      Falling back to simpler configuration...")
        # Simpler fallback without known categoricals
        training_ds = TimeSeriesDataSet(
            df[df.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="Price_per_kg",
            group_ids=["group_id"],
            max_encoder_length=TFT_CFG["max_encoder_length"],
            max_prediction_length=TFT_CFG["max_prediction_length"],
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=["group_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

    # Validation dataset
    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, df, predict=True, stop_randomization=True
    )

    train_loader = training_ds.to_dataloader(
        train=True, batch_size=TFT_CFG["batch_size"], num_workers=0
    )
    val_loader = validation_ds.to_dataloader(
        train=False, batch_size=TFT_CFG["batch_size"] * 2, num_workers=0
    )

    print(f"      Train batches: {len(train_loader)}")
    print(f"      Val batches: {len(val_loader)}")

    # Build TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=TFT_CFG["learning_rate"],
        hidden_size=TFT_CFG["hidden_size"],
        attention_head_size=TFT_CFG["attention_head_size"],
        dropout=TFT_CFG["dropout"],
        hidden_continuous_size=TFT_CFG["hidden_size"] // 2,
        lstm_layers=TFT_CFG["lstm_layers"],
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
        log_interval=10,
        log_val_interval=1,
    )

    n_params = sum(p.numel() for p in tft.parameters())
    print(f"      TFT parameters: {n_params:,}")

    # Training
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, mode="min"),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=TFT_CFG["max_epochs"],
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=TFT_CFG["gradient_clip_val"],
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )

    print(f"      Training TFT for max {TFT_CFG['max_epochs']} epochs on {accelerator}...")
    t0 = time.time()
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_time = time.time() - t0

    print(f"      TFT training complete in {train_time:.1f}s")

    # Save checkpoint
    ckpt_path = MODELS_DIR / "tft_best.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    print(f"      Saved TFT checkpoint: {ckpt_path}")

    # Evaluate on validation set
    print("      Evaluating TFT forecasts...")
    predictions = tft.predict(val_loader, return_x=False, mode="prediction")

    # Get actual values for comparison
    actuals = torch.cat([y[0] for x, y in iter(val_loader)])

    # Calculate RMSE on median quantile
    if predictions.dim() == 3:
        # Shape: (n_samples, prediction_length, n_quantiles)
        median_preds = predictions[:, :, 3]  # median quantile
    else:
        median_preds = predictions

    min_samples = min(len(actuals), len(median_preds))
    actuals_np = actuals[:min_samples].cpu().numpy().flatten()
    preds_np = median_preds[:min_samples].cpu().numpy().flatten()

    min_len_eval = min(len(actuals_np), len(preds_np))
    actuals_np = actuals_np[:min_len_eval]
    preds_np = preds_np[:min_len_eval]

    rmse = float(np.sqrt(np.mean((actuals_np - preds_np) ** 2)))
    mae = float(np.mean(np.abs(actuals_np - preds_np)))
    nonzero = actuals_np != 0
    mape = float(np.mean(np.abs((actuals_np[nonzero] - preds_np[nonzero]) / actuals_np[nonzero])) * 100) if nonzero.any() else 0

    tft_metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "training_time_sec": round(train_time, 1),
        "n_parameters": n_params,
        "device": device_name,
        "epochs_trained": trainer.current_epoch + 1,
    }

    print(f"      TFT Forecast RMSE: {rmse:.4f}")
    print(f"      TFT Forecast MAE:  {mae:.4f}")
    print(f"      TFT Forecast MAPE: {mape:.2f}%")

    # Save metrics
    (MODELS_DIR / "tft_metrics.json").write_text(json.dumps(tft_metrics, indent=2))

    return tft_metrics


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: AUTO-LABEL GENERATION (Stage 2 — Arbitrage Math)
# ──────────────────────────────────────────────────────────────────────────────

def generate_training_labels(df_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Use realized future prices to determine what the optimal action WOULD
    HAVE BEEN at each historical point. This creates ground-truth labels
    for the CatBoost classifier.

    For each record at time t, we look at actual prices at t+3 and t+7
    and compute net profits for SELL_NOW, TRAVEL, and STORE scenarios,
    then label with whichever had the highest profit.
    """
    print_step(4, "Generating ground-truth labels via arbitrage math...")

    df = df_enriched.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Mandi_Name", "Crop", "Date"])

    labels = []
    feature_rows = []
    groups = df.groupby(["Mandi_Name", "Crop"])
    total_groups = len(groups)
    done = 0

    for (mandi, crop), group in groups:
        done += 1
        group = group.sort_values("Date").reset_index(drop=True)

        if len(group) < 30:
            continue

        lam = PERISHABILITY.get(crop, 0.02)
        distance = group["Distance_km"].median()

        for i in range(len(group) - 7):
            price_now = group.loc[i, "Price_per_kg"]
            price_t3 = group.loc[min(i + 3, len(group) - 1), "Price_per_kg"]
            price_t7 = group.loc[min(i + 7, len(group) - 1), "Price_per_kg"]

            qty = 10.0  # 10 quintals standard reference

            # Scenario 1: SELL_LOCAL_NOW (0 days, no transport)
            profit_sell = qty * 100 * price_now  # gross, no costs

            # Scenario 2: TRAVEL_TO_FAR_MANDI (assume 3 days, with transport)
            qty_t3 = qty * math.exp(-lam * 3)
            transport_cost = distance * 15.0  # simplified cost/km
            storage_t3 = 3 * qty * 50.0
            spoilage_t3 = (qty - qty_t3) * 100 * price_t3
            profit_travel = qty_t3 * 100 * price_t3 - transport_cost - storage_t3 - spoilage_t3

            # Scenario 3: STORE_FOR_T_DAYS (7 days, no transport)
            qty_t7 = qty * math.exp(-lam * 7)
            storage_t7 = 7 * qty * 50.0
            spoilage_t7 = (qty - qty_t7) * 100 * price_t7
            profit_store = qty_t7 * 100 * price_t7 - storage_t7 - spoilage_t7

            # Best action
            profits = {
                "SELL_LOCAL_NOW": profit_sell,
                "TRAVEL_TO_FAR_MANDI": profit_travel,
                "STORE_FOR_T_DAYS": profit_store,
            }
            best_action = max(profits, key=profits.get)

            # Feature row for CatBoost
            row = group.iloc[i]
            feature_row = {
                "mandi_id": mandi,
                "crop_name": crop,
                "distance_km": distance,
                "price_now": price_now,
                "price_t3": price_t3,
                "price_t7": price_t7,
                "qty_available": qty,
                # Prophet features
                "prophet_trend_last": row.get("trend", 0),
                "prophet_weekly_last": row.get("weekly", 0),
                "prophet_yearly_last": row.get("yearly", 0),
                # Arbitrage signals
                "net_profit_sell": round(profit_sell, 2),
                "net_profit_travel": round(profit_travel, 2),
                "net_profit_store": round(profit_store, 2),
                "price_momentum_3d": price_t3 - price_now,
                "price_momentum_7d": price_t7 - price_now,
                "profit_ratio_t7_t3": profit_store / (profit_travel + 1e-9),
                # Context
                "perishability_decay_rate": lam,
                "is_rabi": row.get("Is_Rabi_Season", 0),
                "is_kharif": row.get("Is_Kharif_Season", 0),
                "is_weekend": row.get("Is_Weekend", 0),
                "day_of_week": pd.to_datetime(row["Date"]).dayofweek,
                "month": pd.to_datetime(row["Date"]).month,
            }

            feature_rows.append(feature_row)
            labels.append(best_action)

        if done % 5 == 0 or done == total_groups:
            print(f"      [{done}/{total_groups}] {mandi}-{crop}: {len(feature_rows)} samples so far")

    df_features = pd.DataFrame(feature_rows)
    df_features["label"] = labels

    # Distribution
    dist = df_features["label"].value_counts()
    print(f"\n      Label distribution:")
    for label, count in dist.items():
        print(f"        {label}: {count} ({count / len(df_features) * 100:.1f}%)")

    print(f"      Total training samples: {len(df_features):,}")

    return df_features


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: CATBOOST CLASSIFIER (Stage 3)
# ──────────────────────────────────────────────────────────────────────────────

def train_catboost(df_labelled: pd.DataFrame) -> Dict:
    """Train CatBoost 3-class classifier for optimal action prediction."""
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_recall_fscore_support, accuracy_score
    )

    print_step(5, "Training CatBoost decision classifier...")

    # Feature columns
    feature_cols = [
        "mandi_id", "crop_name", "distance_km", "price_now",
        "prophet_trend_last", "prophet_weekly_last", "prophet_yearly_last",
        "price_momentum_3d", "price_momentum_7d",
        "net_profit_sell", "net_profit_travel", "net_profit_store",
        "profit_ratio_t7_t3",
        "perishability_decay_rate", "qty_available",
        "is_rabi", "is_kharif", "is_weekend", "day_of_week", "month",
    ]

    cat_features = ["mandi_id", "crop_name"]

    X = df_labelled[feature_cols].copy()
    y = df_labelled["label"].copy()

    # Time-based split (80/20)
    split = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    print(f"      Train: {len(X_train):,}  Val: {len(X_val):,}")

    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    train_pool = Pool(X_train, label=y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_indices)

    model = CatBoostClassifier(
        iterations=CATBOOST_CFG["iterations"],
        learning_rate=CATBOOST_CFG["learning_rate"],
        depth=CATBOOST_CFG["depth"],
        l2_leaf_reg=CATBOOST_CFG["l2_leaf_reg"],
        random_seed=CATBOOST_CFG["random_seed"],
        eval_metric="MultiClass",
        loss_function="MultiClass",
        verbose=CATBOOST_CFG["verbose"],
        early_stopping_rounds=CATBOOST_CFG["early_stopping_rounds"],
        class_names=["SELL_LOCAL_NOW", "TRAVEL_TO_FAR_MANDI", "STORE_FOR_T_DAYS"],
    )

    t0 = time.time()
    model.fit(train_pool, eval_set=val_pool)
    train_time = time.time() - t0

    # Save model
    model_path = MODELS_DIR / "catboost_classifier.cbm"
    model.save_model(str(model_path))
    print(f"      Model saved: {model_path}")

    # Evaluate
    y_pred = model.predict(val_pool).flatten()
    y_proba = model.predict_proba(val_pool)

    accuracy = float(accuracy_score(y_val, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_pred, average="weighted", zero_division=0
    )

    print(f"\n      Accuracy:  {accuracy:.4f}")
    print(f"      Precision: {float(precision):.4f}")
    print(f"      Recall:    {float(recall):.4f}")
    print(f"      F1-Score:  {float(f1):.4f}")

    # Per-class report
    report = classification_report(y_val, y_pred, zero_division=0)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    labels = sorted(y_val.unique())
    cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels])
    print("      Confusion Matrix:")
    print(cm_df.to_string())

    # Feature importance
    importances = model.get_feature_importance(train_pool)
    fi_df = (pd.DataFrame({"feature": feature_cols, "importance": importances})
             .sort_values("importance", ascending=False))
    print("\n      Top-10 Feature Importances:")
    for _, row in fi_df.head(10).iterrows():
        bar = "#" * int(row["importance"] / 2)
        print(f"        {row['feature']:<28s} {row['importance']:6.2f} {bar}")

    catboost_metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "training_time_sec": round(train_time, 1),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "trained_at": str(datetime.now()),
    }
    (MODELS_DIR / "catboost_metrics.json").write_text(json.dumps(catboost_metrics, indent=2))

    return catboost_metrics


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: COMPARISON WITH XGBOOST BASELINE
# ──────────────────────────────────────────────────────────────────────────────

def compare_with_xgboost(df_enriched: pd.DataFrame, tft_metrics: Dict) -> None:
    """Compare multi-stage pipeline results with existing XGBoost baseline."""
    print_step(6, "Comparing with XGBoost baseline...")

    # Load existing XGBoost metrics
    xgb_models = list(MODELS_DIR.glob("*.meta.json"))
    xgb_metrics = []
    for f in xgb_models:
        if "prophet" not in f.name and "catboost" not in f.name and "tft" not in f.name:
            try:
                m = json.loads(f.read_text())
                xgb_metrics.append(m)
            except Exception:
                pass

    if not xgb_metrics:
        print("      No XGBoost baseline metrics found, skipping comparison.")
        return

    xgb_avg_rmse = np.mean([m.get("rmse", 0) for m in xgb_metrics])
    xgb_avg_mae = np.mean([m.get("mae", 0) for m in xgb_metrics])
    xgb_avg_mape = np.mean([m.get("mape", 0) for m in xgb_metrics])

    tft_rmse = tft_metrics.get("rmse", 0)
    tft_mae = tft_metrics.get("mae", 0)
    tft_mape = tft_metrics.get("mape", 0)

    print(f"\n      {'Metric':<15} {'XGBoost':>12} {'Prophet+TFT':>12} {'Improvement':>12}")
    print(f"      {'-' * 51}")

    for name, xgb_val, tft_val in [
        ("RMSE", xgb_avg_rmse, tft_rmse),
        ("MAE", xgb_avg_mae, tft_mae),
        ("MAPE %", xgb_avg_mape, tft_mape),
    ]:
        if xgb_val > 0:
            improvement = ((xgb_val - tft_val) / xgb_val) * 100
            arrow = "BETTER" if improvement > 0 else "WORSE"
            print(f"      {name:<15} {xgb_val:>12.4f} {tft_val:>12.4f} {improvement:>+10.1f}% {arrow}")
        else:
            print(f"      {name:<15} {xgb_val:>12.4f} {tft_val:>12.4f}      N/A")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    print_header(f"MULTI-STAGE PIPELINE TRAINING: Prophet -> TFT -> CatBoost")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models dir: {MODELS_DIR}")
    print()

    # Step 1: Load data
    df = load_and_process_data()

    # Step 2: Prophet trend decomposition
    df_enriched, prophet_models = train_prophet_all(df)

    # Save enriched data for later use
    enriched_path = "ml_arbitrage/processed_data_with_prophet.csv"
    df_enriched.to_csv(enriched_path, index=False)
    print(f"      Saved enriched data: {enriched_path}")

    # Step 3: TFT forecasting
    tft_metrics = train_tft(df_enriched)

    # Step 4: Generate CatBoost training labels
    df_labelled = generate_training_labels(df_enriched)

    # Save labelled data
    labelled_path = "ml_arbitrage/catboost_training_data.csv"
    df_labelled.to_csv(labelled_path, index=False)
    print(f"      Saved labelled data: {labelled_path}")

    # Step 5: Train CatBoost classifier
    catboost_metrics = train_catboost(df_labelled)

    # Step 6: Compare with XGBoost baseline
    if isinstance(tft_metrics, dict) and "error" not in tft_metrics:
        compare_with_xgboost(df_enriched, tft_metrics)

    # ── Final Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print_header("TRAINING COMPLETE")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)\n")
    print(f"  Models saved to: {MODELS_DIR}/")
    print(f"    - Prophet:  prophet_models_meta.json")
    print(f"    - TFT:      tft_best.ckpt + tft_metrics.json")
    print(f"    - CatBoost: catboost_classifier.cbm + catboost_metrics.json")
    print()

    if isinstance(tft_metrics, dict) and "error" not in tft_metrics:
        print(f"  TFT Forecasting Performance:")
        print(f"    RMSE: {tft_metrics.get('rmse', '?')}")
        print(f"    MAE:  {tft_metrics.get('mae', '?')}")
        print(f"    MAPE: {tft_metrics.get('mape', '?')}%")
        print()

    print(f"  CatBoost Classification Performance:")
    print(f"    Accuracy:  {catboost_metrics.get('accuracy', '?')}")
    print(f"    F1-Score:  {catboost_metrics.get('f1_score', '?')}")
    print()

    print(f"  Pipeline Architecture:")
    print(f"    Raw Prices --> [Prophet] --> trend/weekly/yearly")
    print(f"                       |")
    print(f"                       v")
    print(f"    Prophet + Prices --> [TFT] --> price_t+3, price_t+7")
    print(f"                           |")
    print(f"                           v")
    print(f"    Forecasts --> [Arbitrage Math] --> net_profit scenarios")
    print(f"                           |")
    print(f"                           v")
    print(f"    All features --> [CatBoost] --> SELL / TRAVEL / STORE")
    print()
    print(f"  Next: python get_recommendation.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
