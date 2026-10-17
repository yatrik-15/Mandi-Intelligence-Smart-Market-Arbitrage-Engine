"""
features.py — Leakage-free features for direct multi-horizon forecasting.

Framing
-------
For every day `t` on which a mandi reported a price p_t (the forecast origin)
and every horizon h = 1..MAX_HORIZON we predict

    y = log(p_{t+h} / p_t)

using only information available at the end of day t. The target is a relative
change, so one model per crop can be pooled across all mandis (Ahmedabad at
₹30/kg and Dahod at ₹12/kg share the same dynamics) and works for mandis that
only report a few days a week.

All rolling windows are trailing and include day t itself, which is fine: p_t is
known at the origin; the target lies strictly in the future.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

MAX_HORIZON = 14          # days; covers "price is k days stale" + "wait up to 7 days"
FFILL_LIMIT = 14          # carry the last report forward at most this many days

FEATURE_COLS: List[str] = [
    "horizon",
    "log_price",
    "dev_ma_7", "dev_ma_14", "dev_ma_30", "dev_ma_90", "dev_ma_365",
    "ret_1", "ret_3", "ret_7", "ret_14", "ret_30",
    "vol_14", "vol_60",
    "n_reports_14", "days_since_prev",
    "premium_vs_state", "state_ret_7", "state_ret_30", "state_dev_ma_30",
    "seasonal_fwd", "yoy_level",
    "doy_sin", "doy_cos",
]


def _state_index(panel: pd.DataFrame) -> pd.Series:
    """Daily median log price across all mandis of one crop, forward-filled."""
    s = np.log(panel["Price_per_kg"]).groupby(panel["Date"]).median()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full).ffill(limit=FFILL_LIMIT)


def _series_frame(dates: pd.Series, prices: pd.Series, state: pd.Series) -> pd.DataFrame:
    """Daily-calendar features for one (mandi, crop) series."""
    idx = pd.date_range(dates.min(), dates.max(), freq="D")
    lp = pd.Series(np.log(prices.to_numpy()), index=pd.DatetimeIndex(dates)).reindex(idx)
    lff = lp.ffill(limit=FFILL_LIMIT)
    obs = lp.notna()

    f = pd.DataFrame(index=idx)
    f["lp"] = lp
    f["log_price"] = lff
    for w in (7, 14, 30, 90, 365):
        f[f"dev_ma_{w}"] = lff - lp.rolling(w, min_periods=1).mean()
    for k in (1, 3, 7, 14, 30):
        f[f"ret_{k}"] = lff - lff.shift(k)
    d1 = lff.diff()
    f["vol_14"] = d1.rolling(14, min_periods=3).std()
    f["vol_60"] = d1.rolling(60, min_periods=10).std()
    f["n_reports_14"] = obs.rolling(14, min_periods=1).sum()
    obs_day = pd.Series(np.where(obs, np.arange(len(idx)), np.nan), index=idx)
    f["days_since_prev"] = np.arange(len(idx)) - obs_day.shift(1).ffill()

    st = state.reindex(idx)
    f["premium_vs_state"] = lff - st
    f["state_ret_7"] = st - st.shift(7)
    f["state_ret_30"] = st - st.shift(30)
    f["state_dev_ma_30"] = st - st.rolling(30, min_periods=1).mean()

    # Last year's smoothed curve, for the seasonal_fwd feature (horizon-specific)
    f["_smooth"] = lff.rolling(15, center=True, min_periods=3).mean()
    f["yoy_level"] = lff - f["_smooth"].shift(365)

    doy = idx.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f


def build_series_frames(panel: pd.DataFrame) -> dict[Tuple[str, str], pd.DataFrame]:
    """Daily feature frames for every (mandi, crop) series in the panel."""
    frames = {}
    for crop, crop_df in panel.groupby("Crop"):
        state = _state_index(crop_df)
        for mandi, g in crop_df.groupby("Mandi_Name"):
            if len(g) < 2:
                continue
            frames[(mandi, crop)] = _series_frame(g["Date"], g["Price_per_kg"], state)
    return frames


def _expand_horizons(f: pd.DataFrame, origins: np.ndarray,
                     horizons: range, with_target: bool) -> pd.DataFrame:
    """Repeat origin rows for each horizon; add horizon-specific columns."""
    lp = f["lp"].to_numpy()
    smooth = f["_smooth"].to_numpy()
    n = len(f)
    base = f.iloc[origins]
    parts = []
    for h in horizons:
        part = base.copy()
        part["horizon"] = h
        # What last year's prices did over the same h days (both points ≤ t).
        a, b = origins - 365 + h, origins - 365
        valid = b >= 0
        sf = np.full(len(origins), np.nan)
        sf[valid] = smooth[a[valid]] - smooth[b[valid]]
        part["seasonal_fwd"] = sf
        if with_target:
            tgt = origins + h
            y = np.full(len(origins), np.nan)
            inside = tgt < n
            y[inside] = lp[tgt[inside]] - lp[origins[inside]]
            part["y"] = y
        parts.append(part)
    out = pd.concat(parts)
    out["origin_date"] = out.index
    return out.reset_index(drop=True)


def training_rows(frames: dict, max_horizon: int = MAX_HORIZON) -> pd.DataFrame:
    """Supervised rows: every reported day × every horizon with a reported target."""
    rows = []
    for (mandi, crop), f in frames.items():
        origins = np.flatnonzero(f["lp"].notna().to_numpy())
        r = _expand_horizons(f, origins, range(1, max_horizon + 1), with_target=True)
        r = r[r["y"].notna()]
        r["Mandi_Name"], r["Crop"] = mandi, crop
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def latest_rows(frames: dict, max_horizon: int = MAX_HORIZON) -> pd.DataFrame:
    """Inference rows anchored at each series' most recent report."""
    rows = []
    for (mandi, crop), f in frames.items():
        last = np.array([np.flatnonzero(f["lp"].notna().to_numpy())[-1]])
        r = _expand_horizons(f, last, range(1, max_horizon + 1), with_target=False)
        r["Mandi_Name"], r["Crop"] = mandi, crop
        r["last_price"] = float(np.exp(f["lp"].iloc[last[0]]))
        rows.append(r)
    return pd.concat(rows, ignore_index=True)
