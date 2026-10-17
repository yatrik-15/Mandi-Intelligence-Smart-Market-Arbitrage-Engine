"""
data_loader.py — Load and clean AGMARKNET mandi price data.

Input : any AGMARKNET / data.gov.in export (see scripts/fetch_agmarknet.py)
        with Arrival_Date, State, Market, Commodity and Modal_Price columns.
Output: one row per (Date, Mandi_Name, Crop) with the median modal price in ₹/kg.

Cleaning steps
--------------
1. Normalise column names (data.gov.in uses `Modal_x0020_Price`).
2. Keep the target state and crops; parse dd/mm/yyyy dates.
3. Canonicalise market names: "Ahmedabad(Chimanbhai Patal Market Vasana) APMC"
   and "Ahmedabad" are the same mandi.
4. Aggregate varieties / sub-yards to one median price per mandi-crop-day.
5. Drop data-entry errors: prices outside ₹1–₹250/kg, and points more than
   2.5× away from the series' local median (real spikes move the median too,
   so they survive; isolated typos do not).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_CROPS: List[str] = ["Onion", "Tomato", "Potato"]

_COLUMN_ALIASES = {
    "arrival_date": "Arrival_Date", "date": "Arrival_Date",
    "state": "State", "district": "District",
    "market": "Market", "commodity": "Commodity",
    "modal_price": "Modal_Price", "min_price": "Min_Price", "max_price": "Max_Price",
}

# Spelling variants that refer to the same market town.
_MANDI_ALIASES = {
    "nadiyad": "Nadiad",
    "vankaner": "Wankaner",
    "talalagir": "Talala",
    "mahesana": "Mehsana",
    "baroda": "Vadodara",
    "wadhwan": "Vadhvan",
    "siddhpur": "Sidhpur",
}

MIN_PRICE_PER_KG = 1.0
MAX_PRICE_PER_KG = 250.0
OUTLIER_RATIO = 2.5          # vs. local rolling median
OUTLIER_WINDOW = 15          # observations


def clean_mandi_name(raw: object) -> Optional[str]:
    """'Rajkot(Veg.Sub Yard) APMC' → 'Rajkot'."""
    if not isinstance(raw, str):
        return None
    name = raw.split("(")[0]
    name = re.sub(r"\bAPMC\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" ,.-")
    if not name:
        return None
    return _MANDI_ALIASES.get(name.lower(), name.title())


def load_prices(
    csv_paths: Iterable[str | Path],
    state: str = "Gujarat",
    crops: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load one or more raw CSVs and return a clean daily panel with columns
    Date, Mandi_Name, Crop, District, Price_per_kg, N_Reports.
    """
    crops = crops or TARGET_CROPS
    frames = []
    for path in csv_paths:
        path = Path(path)
        if not path.exists():
            continue
        raw = pd.read_csv(path, low_memory=False)
        raw.columns = [_COLUMN_ALIASES.get(c.replace("_x0020_", "_").strip().lower(), c)
                       for c in raw.columns]
        frames.append(raw)
        logger.info("Loaded %s rows from %s", f"{len(raw):,}", path.name)
    if not frames:
        raise FileNotFoundError(f"None of the dataset files exist: {list(map(str, csv_paths))}")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["State"].astype(str).str.strip().str.casefold() == state.casefold()]
    df = df[df["Commodity"].isin(crops)]

    df = pd.DataFrame({
        "Date": pd.to_datetime(df["Arrival_Date"], format="%d/%m/%Y", errors="coerce"),
        "Mandi_Name": df["Market"].map(clean_mandi_name),
        "Crop": df["Commodity"],
        "District": df.get("District", pd.Series("Unknown", index=df.index)).astype(str).str.strip(),
        "Price_per_kg": pd.to_numeric(df["Modal_Price"], errors="coerce") / 100.0,
    }).dropna(subset=["Date", "Mandi_Name", "Price_per_kg"])

    df = df[df["Price_per_kg"].between(MIN_PRICE_PER_KG, MAX_PRICE_PER_KG)]

    df = (df.groupby(["Mandi_Name", "Crop", "Date"], as_index=False)
            .agg(Price_per_kg=("Price_per_kg", "median"),
                 District=("District", "first"),
                 N_Reports=("Price_per_kg", "size")))

    df = _drop_isolated_outliers(df)
    df = df.sort_values(["Crop", "Mandi_Name", "Date"]).reset_index(drop=True)
    logger.info("Clean panel: %s rows, %d mandis, %s → %s",
                f"{len(df):,}", df["Mandi_Name"].nunique(),
                df["Date"].min().date(), df["Date"].max().date())
    return df


def _drop_isolated_outliers(df: pd.DataFrame) -> pd.DataFrame:
    log_p = np.log(df["Price_per_kg"])
    local_median = (log_p.groupby([df["Mandi_Name"], df["Crop"]])
                         .transform(lambda s: s.rolling(OUTLIER_WINDOW, center=True,
                                                        min_periods=3).median()))
    keep = (log_p - local_median).abs().fillna(0) <= np.log(OUTLIER_RATIO)
    if (~keep).any():
        logger.info("Dropped %d isolated price outliers", int((~keep).sum()))
    return df[keep]


def active_series(
    df: pd.DataFrame,
    min_obs: int = 150,
    max_staleness_days: int = 30,
) -> pd.DataFrame:
    """
    (Mandi_Name, Crop) pairs with enough history and a recent report, i.e. the
    mandis we can honestly recommend today.
    """
    latest_per_crop = df.groupby("Crop")["Date"].transform("max")
    stats = (df.assign(_crop_latest=latest_per_crop)
               .groupby(["Mandi_Name", "Crop"])
               .agg(n_obs=("Date", "size"), last_date=("Date", "max"),
                    crop_latest=("_crop_latest", "first")))
    stale = (stats["crop_latest"] - stats["last_date"]).dt.days
    return stats[(stats["n_obs"] >= min_obs) & (stale <= max_staleness_days)].reset_index()
