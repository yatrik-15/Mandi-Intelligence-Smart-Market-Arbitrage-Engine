"""
forecaster.py — Direct multi-horizon XGBoost price forecaster (v4).

One XGBoost model per crop, pooled across mandis, predicts log(p_{t+h}/p_t)
for h = 1..14 days directly (no recursive feeding of predictions).

Evaluation protocol (per crop, split by forecast-origin date)
-------------------------------------------------------------
    |──────── train ────────|gap|── validation ──|gap|──── test ────|
                                  early stopping       reported metrics
                                  + interval calib.

The gaps are MAX_HORIZON days, so no target in one block overlaps the next.
Test metrics are compared with two naive baselines evaluated on exactly the
same rows: persistence (p_{t+h} = p_t) and the trailing 7-day average.
After evaluation the model is refit on all data with the chosen number of
trees, so production forecasts use the most recent prices.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import (FEATURE_COLS, MAX_HORIZON, build_series_frames,
                       latest_rows, training_rows)

logger = logging.getLogger(__name__)

XGB_PARAMS: Dict = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "objective": "reg:pseudohubererror",
    "huber_slope": 0.1,
    "tree_method": "hist",
    "early_stopping_rounds": 150,
    "random_state": 42,
    "n_jobs": -1,
}
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
INTERVAL = (0.10, 0.90)          # 80 % prediction interval
REPORT_HORIZONS = (1, 3, 7, 14)


def _price_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - actual
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / actual) * 100),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


class PriceForecaster:
    """Train, evaluate, persist and serve per-crop pooled forecasters."""

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.meta: Dict[str, Dict] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, panel: pd.DataFrame, crops: Optional[List[str]] = None) -> Dict[str, Dict]:
        frames = build_series_frames(panel)
        rows = training_rows(frames)
        reports = {}
        for crop in crops or sorted(rows["Crop"].unique()):
            crop_rows = rows[rows["Crop"] == crop]
            reports[crop] = self._fit_crop(crop, crop_rows,
                                           data_end=panel.loc[panel["Crop"] == crop, "Date"].max())
        return reports

    def _fit_crop(self, crop: str, rows: pd.DataFrame, data_end: pd.Timestamp) -> Dict:
        dates = rows["origin_date"]
        start, end = dates.min(), dates.max()
        span = end - start
        gap = pd.Timedelta(days=MAX_HORIZON)
        test_start = end - span * TEST_FRACTION
        val_start = test_start - gap - span * VAL_FRACTION

        train = rows[dates < val_start - gap]
        val = rows[(dates >= val_start) & (dates < test_start - gap)]
        test = rows[dates >= test_start]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(train[FEATURE_COLS], train["y"],
                  eval_set=[(val[FEATURE_COLS], val["y"])], verbose=False)
        n_trees = model.best_iteration + 1

        # Interval calibration on validation residuals, per horizon (log space)
        val_resid = val["y"].to_numpy() - model.predict(val[FEATURE_COLS])
        quantiles = {}
        for h in range(1, MAX_HORIZON + 1):
            r = val_resid[val["horizon"].to_numpy() == h]
            lo, hi = np.quantile(r, INTERVAL) if len(r) >= 30 else np.quantile(val_resid, INTERVAL)
            quantiles[str(h)] = [float(lo), float(hi)]

        metrics = self._evaluate(model, test, quantiles)
        metrics["split"] = {
            "train": [str(train["origin_date"].min().date()), str(train["origin_date"].max().date()), len(train)],
            "validation": [str(val["origin_date"].min().date()), str(val["origin_date"].max().date()), len(val)],
            "test": [str(test["origin_date"].min().date()), str(test["origin_date"].max().date()), len(test)],
        }

        # Refit on everything for production
        final_params = {**XGB_PARAMS, "n_estimators": int(n_trees * 1.1)}
        final_params.pop("early_stopping_rounds")
        final = xgb.XGBRegressor(**final_params)
        final.fit(rows[FEATURE_COLS], rows["y"], verbose=False)

        self.models[crop] = final
        self.meta[crop] = {
            "crop": crop,
            "feature_cols": FEATURE_COLS,
            "max_horizon": MAX_HORIZON,
            "n_trees": int(n_trees * 1.1),
            "residual_quantiles": quantiles,
            "data_end": str(pd.Timestamp(data_end).date()),
            "n_series": int(rows.groupby(["Mandi_Name"]).ngroups),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics,
        }
        logger.info("%s: %d trees, test MAE@7d ₹%.2f vs persistence ₹%.2f", crop, n_trees,
                    metrics["by_horizon"]["7"]["model"]["mae"],
                    metrics["by_horizon"]["7"]["persistence"]["mae"])
        return self.meta[crop]

    @staticmethod
    def _evaluate(model: xgb.XGBRegressor, test: pd.DataFrame, quantiles: Dict) -> Dict:
        p0 = np.exp(test["log_price"].to_numpy())
        actual = p0 * np.exp(test["y"].to_numpy())
        y_hat = model.predict(test[FEATURE_COLS])
        preds = {
            "model": p0 * np.exp(y_hat),
            "persistence": p0,
            "moving_avg_7": p0 * np.exp(-test["dev_ma_7"].to_numpy()),
        }
        h = test["horizon"].to_numpy()
        lo = np.array([quantiles[str(k)][0] for k in h])
        hi = np.array([quantiles[str(k)][1] for k in h])
        inside = (actual >= preds["model"] * np.exp(lo)) & (actual <= preds["model"] * np.exp(hi))

        def block(mask):
            return {name: _price_metrics(actual[mask], p[mask]) for name, p in preds.items()}

        by_h = {}
        for k in REPORT_HORIZONS:
            m = h == k
            if m.any():
                by_h[str(k)] = {**block(m), "interval_coverage": float(inside[m].mean()),
                                "n": int(m.sum())}
        return {"overall": {**block(np.ones_like(h, dtype=bool)),
                            "interval_coverage": float(inside.mean()), "n": int(len(h))},
                "by_horizon": by_h}

    # ── Inference ─────────────────────────────────────────────────────────────

    def forecast(
        self,
        panel: pd.DataFrame,
        crop: str,
        days_ahead: int = 7,
        max_staleness_days: int = 7,
        mandis: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Price path for day 0 (today) .. `days_ahead` at every mandi.

        "Today" is the crop's latest report date in the panel. For a mandi whose
        last report is g days old, day d is forecast at horizon g + d; day 0 is
        the observed price only when g = 0. Mandis staler than
        `max_staleness_days` are skipped.
        """
        model, meta = self.models[crop], self.meta[crop]
        crop_panel = panel[panel["Crop"] == crop]
        today = crop_panel["Date"].max()

        frames = build_series_frames(crop_panel)
        rows = latest_rows(frames)
        if mandis is not None:   # state-level features still use every mandi
            rows = rows[rows["Mandi_Name"].isin(mandis)]
        rows["staleness"] = (today - rows["origin_date"]).dt.days
        rows = rows[rows["staleness"] <= max_staleness_days]
        rows["Day_Ahead"] = rows["horizon"] - rows["staleness"]
        rows = rows[rows["Day_Ahead"].between(0, days_ahead)]

        y_hat = model.predict(rows[FEATURE_COLS])
        q = meta["residual_quantiles"]
        lo = np.array([q[str(h)][0] for h in rows["horizon"]])
        hi = np.array([q[str(h)][1] for h in rows["horizon"]])
        point = rows["last_price"].to_numpy() * np.exp(y_hat)

        out = pd.DataFrame({
            "Mandi_Name": rows["Mandi_Name"].to_numpy(),
            "Crop": crop,
            "Day_Ahead": rows["Day_Ahead"].to_numpy(),
            "Date": (today + pd.to_timedelta(rows["Day_Ahead"], unit="D")).to_numpy(),
            "Predicted_Price": point,
            "Lower_Bound": point * np.exp(lo),
            "Upper_Bound": point * np.exp(hi),
            "Is_Observed": False,
        })

        # Day 0 for mandis that reported today: the observed price, no uncertainty
        fresh = (rows.drop_duplicates("Mandi_Name").query("staleness == 0"))
        observed = pd.DataFrame({
            "Mandi_Name": fresh["Mandi_Name"].to_numpy(), "Crop": crop, "Day_Ahead": 0,
            "Date": today, "Predicted_Price": fresh["last_price"].to_numpy(),
            "Lower_Bound": fresh["last_price"].to_numpy(),
            "Upper_Bound": fresh["last_price"].to_numpy(), "Is_Observed": True,
        })
        out = pd.concat([observed, out], ignore_index=True)
        return out.sort_values(["Mandi_Name", "Day_Ahead"]).reset_index(drop=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for crop, model in self.models.items():
            model.save_model(str(self.models_dir / f"{crop}.ubj"))
            (self.models_dir / f"{crop}.meta.json").write_text(
                json.dumps(self.meta[crop], indent=2), encoding="utf-8")

    def load(self, crops: List[str]) -> bool:
        """Load saved models; returns False if any crop is missing or outdated."""
        for crop in crops:
            mpath = self.models_dir / f"{crop}.ubj"
            jpath = self.models_dir / f"{crop}.meta.json"
            if not (mpath.exists() and jpath.exists()):
                return False
            meta = json.loads(jpath.read_text(encoding="utf-8"))
            if meta.get("feature_cols") != FEATURE_COLS:
                return False
            model = xgb.XGBRegressor()
            model.load_model(str(mpath))
            self.models[crop], self.meta[crop] = model, meta
        return True

    def report(self) -> List[Dict]:
        return [{"crop": c, "data_end": m["data_end"], "n_series": m["n_series"],
                 "trained_at": m["trained_at"], "metrics": m["metrics"]}
                for c, m in self.meta.items()]
