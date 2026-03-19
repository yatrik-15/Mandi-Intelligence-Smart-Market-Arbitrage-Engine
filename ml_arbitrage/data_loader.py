"""
ml_arbitrage/data_loader.py — Refined Data Pipeline v3.0
==========================================================
FIXES FROM v2.x
---------------
  ❌ BUG: Column rename using '_x0020_' only matched one format — broke on raw CSVs.
  ❌ BUG: `filter_and_process()` filtered by exact mandi names that don't exist in
          the Kaggle dataset (Ahmedabad, Mehsana are rarely present) → 0–7 rows.
  ❌ BUG: Price outlier records (₹0, ₹99999) were passed straight into XGBoost.
  ❌ BUG: `_calculate_traffic_score` used `np.random` without seeding → non-deterministic.
  ❌ BUG: `filter_and_process(days=None)` didn't propagate the full date range properly.
  ❌ MISSING: No data schema validation; silent failures downstream.
  ❌ MISSING: No duplicate-row deduplication before feature engineering.

NEW IN v3.0
-----------
  ✅ Schema-agnostic column detection — handles Kaggle, eNAM, AGMARKNET, and custom CSVs.
  ✅ Removes mandi-name hard-filter; uses ALL Gujarat mandis automatically.
  ✅ IQR-based price outlier winsorisation per crop.
  ✅ Deterministic traffic score (hash-based seed, no random state leakage).
  ✅ Full data validation report returned alongside DataFrame.
  ✅ Duplicate-date deduplication with modal-price aggregation.
  ✅ Supports multiple date formats with auto-detection.
  ✅ Adds harvest-season indicator for Rabi/Kharif cycles.

RECOMMENDED DATASETS
--------------------
  1. AGMARKNET (Primary — government source)
       URL    : https://agmarknet.gov.in/SearchCmmMkt.aspx
       Format : CSV download, daily modal prices
       Fields : Date, State, District, Market, Commodity, Min/Max/Modal Price

  2. Kaggle "Daily Wholesale Commodity Prices India Mandis"
       URL    : https://www.kaggle.com/datasets/ramkrijal/agriculture-vegetables-fruits-time-series-prices
       Format : commodity_price.csv  (2+ years, ~2,700 records for Gujarat)

  3. eNAM API (Real-time, best for production)
       URL    : https://enam.gov.in/web/
       Auth   : API key required (free for dev)
       Fields : JSON, live prices every 15 min during market hours

  4. data.gov.in Horticulture Statistics
       URL    : https://data.gov.in/catalog/wholesale-price-index-horticulture-commodities
       Format : CSV/XLSX

  5. Gujarat APMC Portal
       URL    : https://agri.gujarat.gov.in/
       Format : Manual download / screen scraping

  For Testing (Minimal Viable):
       Use the included `generate_rich_synthetic_data()` which produces a
       statistically realistic 2-year panel for 8 Gujarat mandis × 5 crops.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Schema detection helpers
# ──────────────────────────────────────────────────────────────────────────────

# Priority-ordered candidate column names (lowercase) for each semantic field.
_SCHEMA_MAP: Dict[str, List[str]] = {
    "date":         ["arrival_date", "date", "trade_date", "price_date", "mandi_date"],
    "state":        ["state", "state_name"],
    "district":     ["district", "district_name", "dist"],
    "market":       ["market", "market_name", "mandi", "mandi_name", "apmc"],
    "commodity":    ["commodity", "commodity_name", "crop", "item"],
    "modal_price":  ["modal_price", "modal price", "price", "modal", "modalprice"],
    "min_price":    ["min_price", "min price", "minimum_price", "minprice"],
    "max_price":    ["max_price", "max price", "maximum_price", "maxprice"],
}

_DATE_FORMATS = [
    "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y",
    "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%d-%b-%Y",
]


def _detect_column(df: pd.DataFrame, semantic: str) -> Optional[str]:
    """Return the first matching column name for a semantic field."""
    candidates = _SCHEMA_MAP.get(semantic, [])
    col_lower   = {c.lower().replace(" ", "_").replace("_x0020_", "_"): c
                   for c in df.columns}
    for cand in candidates:
        if cand in col_lower:
            return col_lower[cand]
    return None


def _parse_dates(series: pd.Series) -> pd.Series:
    """Try multiple date formats; return NaT for failures."""
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            logger.debug("Date format matched: %s", fmt)
            return parsed
        except (ValueError, TypeError):
            continue
    # Fallback: let pandas infer
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


# ──────────────────────────────────────────────────────────────────────────────
# Mandi configuration
# ──────────────────────────────────────────────────────────────────────────────

# Base distances (km) from Gandhinagar — used only when a mandi is unknown.
# These are realistic road distances, not straight-line.
MANDI_BASE_CONFIG: Dict[str, Dict] = {
    "Ahmedabad":   {"distance_km": 26,  "district": "Ahmedabad"},
    "Mehsana":     {"distance_km": 62,  "district": "Mahesana"},
    "Rajkot":      {"distance_km": 237, "district": "Rajkot"},
    "Surat":       {"distance_km": 273, "district": "Surat"},
    "Anand":       {"distance_km": 98,  "district": "Anand"},
    "Bharuch":     {"distance_km": 211, "district": "Bharuch"},
    "Amreli":      {"distance_km": 277, "district": "Amreli"},
    "Vadodara":    {"distance_km": 100, "district": "Vadodara"},
    "Bhavnagar":   {"distance_km": 200, "district": "Bhavnagar"},
    "Jamnagar":    {"distance_km": 330, "district": "Jamnagar"},
    "Junagadh":    {"distance_km": 330, "district": "Junagadh"},
    "Bilimora":    {"distance_km": 290, "district": "Navsari"},
    "Damnagar":    {"distance_km": 260, "district": "Amreli"},
    "Mansa":       {"distance_km": 55,  "district": "Gandhinagar"},
    "Padra":       {"distance_km": 110, "district": "Vadodara"},
    "Vadhvan":     {"distance_km": 165, "district": "Surendranagar"},
    "Wankaner":    {"distance_km": 220, "district": "Rajkot"},
    "Gondal":      {"distance_km": 265, "district": "Rajkot"},
    "Unjha":       {"distance_km": 90,  "district": "Mahesana"},
    "Sidhpur":     {"distance_km": 105, "district": "Patan"},
}

TARGET_CROPS = ["Onion", "Tomato", "Potato", "Wheat", "Cotton"]

# District → approximate distance from Gandhinagar
DISTRICT_DISTANCES: Dict[str, float] = {
    "Ahmedabad": 30, "Gandhinagar": 5, "Mahesana": 62, "Mehsana": 62,
    "Rajkot": 237, "Surat": 273, "Vadodara": 100, "Anand": 98,
    "Bharuch": 211, "Amreli": 277, "Bhavnagar": 200, "Jamnagar": 330,
    "Junagadh": 330, "Kutch": 370, "Patan": 120, "Porbandar": 400,
    "Sabarkantha": 90, "Surendranagar": 165, "Tapi": 310, "Dang": 195,
    "Narmada": 145, "Navsari": 290, "Valsad": 310,
}


class MandiDataLoader:
    """
    Schema-agnostic loader for Indian agricultural price datasets.

    Supports AGMARKNET, Kaggle commodity_price.csv, eNAM exports,
    and custom CSVs with automatic column detection.
    """

    def __init__(self) -> None:
        self.raw_data: Optional[pd.DataFrame]       = None
        self.processed_data: Optional[pd.DataFrame] = None
        self._validation_report: Dict               = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load raw CSV with automatic encoding detection.

        Parameters
        ----------
        csv_path : str  — path to the CSV file

        Returns
        -------
        pd.DataFrame — raw (unprocessed) data
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}\n"
                                    "See module docstring for recommended dataset URLs.")

        for enc in ("utf-8", "latin-1", "iso-8859-1", "cp1252"):
            try:
                df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
                logger.info("Loaded %d rows from %s (encoding: %s)", len(df), csv_path, enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {csv_path} with any supported encoding.")

        # Normalise column names (strip whitespace, replace URL-encoded spaces)
        df.columns = (df.columns
                      .str.strip()
                      .str.replace("_x0020_", "_", regex=False)
                      .str.replace(r"\s+", "_", regex=True))

        self.raw_data = df
        print(f"✅ Loaded {len(df):,} records  |  Columns: {list(df.columns)}")
        return df

    def filter_and_process(
        self,
        days: Optional[int] = None,
        state_filter: str = "Gujarat",
        target_crops: Optional[List[str]] = None,
        min_price_inr: float = 10.0,
        max_price_inr: float = 50_000.0,
    ) -> pd.DataFrame:
        """
        Clean, filter, and featurise the loaded dataset.

        Parameters
        ----------
        days           : Use last N days only. None = all historical data.
        state_filter   : State to filter (default: Gujarat).
        target_crops   : Crops to keep. None = use TARGET_CROPS constant.
        min_price_inr  : Floor price ₹/quintal for outlier removal.
        max_price_inr  : Ceiling price ₹/quintal for outlier removal.

        Returns
        -------
        pd.DataFrame with columns:
            Date, Mandi_Name, Crop, District, Distance_km,
            Price_per_kg, Traffic_Congestion_Score,
            Is_Rabi_Season, Is_Kharif_Season, Is_Weekend
        """
        if self.raw_data is None:
            raise RuntimeError("Call load_data() before filter_and_process().")

        if target_crops is None:
            target_crops = TARGET_CROPS

        df = self.raw_data.copy()

        # ── 1. Detect semantic columns ─────────────────────────────────────
        col_date     = _detect_column(df, "date")
        col_state    = _detect_column(df, "state")
        col_district = _detect_column(df, "district")
        col_market   = _detect_column(df, "market")
        col_commodity= _detect_column(df, "commodity")
        col_price    = _detect_column(df, "modal_price") or _detect_column(df, "min_price")

        missing = [k for k, v in {
            "date": col_date, "market": col_market,
            "commodity": col_commodity, "price": col_price,
        }.items() if v is None]
        if missing:
            raise ValueError(f"Cannot detect required columns: {missing}\n"
                             f"Available: {list(df.columns)}")

        logger.info("Column mapping → date=%s market=%s commodity=%s price=%s",
                    col_date, col_market, col_commodity, col_price)

        # ── 2. Parse dates ────────────────────────────────────────────────
        df["Date"] = _parse_dates(df[col_date])
        bad_dates  = df["Date"].isna().sum()
        if bad_dates:
            logger.warning("Dropping %d rows with unparseable dates.", bad_dates)
        df = df.dropna(subset=["Date"])

        # ── 3. State filter ───────────────────────────────────────────────
        if col_state and state_filter:
            mask = df[col_state].str.contains(state_filter, case=False, na=False)
            df   = df[mask]
            if df.empty:
                raise ValueError(f"No records found for state='{state_filter}'. "
                                 f"Available states: {self.raw_data[col_state].unique()[:10].tolist()}")

        # ── 4. Crop filter (partial match, case-insensitive) ───────────────
        pattern = "|".join(target_crops)
        df = df[df[col_commodity].str.contains(pattern, case=False, na=False)]
        if df.empty:
            raise ValueError(f"No records match crops: {target_crops}")

        # ── 5. Standardise names ──────────────────────────────────────────
        df["Mandi_Name"] = df[col_market].apply(self._clean_mandi_name)
        df["Crop"]       = df[col_commodity].apply(
            lambda x: self._standardise_crop(x, target_crops)
        )
        df = df.dropna(subset=["Mandi_Name", "Crop"])

        # ── 6. District ───────────────────────────────────────────────────
        df["District"] = (df[col_district].str.strip()
                          if col_district else "Unknown")

        # ── 7. Price cleaning ─────────────────────────────────────────────
        df["Raw_Price"] = pd.to_numeric(df[col_price], errors="coerce")
        df = df.dropna(subset=["Raw_Price"])
        df = df[(df["Raw_Price"] >= min_price_inr) & (df["Raw_Price"] <= max_price_inr)]

        # IQR winsorisation per crop (removes fat-tail outliers)
        df = self._winsorise_prices(df, "Raw_Price")

        # Convert ₹/quintal → ₹/kg  (1 quintal = 100 kg)
        df["Price_per_kg"] = (df["Raw_Price"] / 100).round(4)

        # ── 8. Date range filter ──────────────────────────────────────────
        end_date = df["Date"].max()
        if days is not None:
            start_date = end_date - timedelta(days=days)
            df = df[df["Date"] >= start_date]
            logger.info("Date filter: last %d days (%s → %s)", days, start_date.date(), end_date.date())

        # ── 9. Deduplicate (same date + mandi + crop) ─────────────────────
        before = len(df)
        df = (df.groupby(["Date", "Mandi_Name", "Crop"], as_index=False)
                .agg({"Price_per_kg": "median", "District": "first", "Raw_Price": "median"}))
        logger.info("Deduplication: %d → %d rows", before, len(df))

        # ── 10. Add engineered features ───────────────────────────────────
        df = self._add_features(df)

        # ── 11. Validation report ─────────────────────────────────────────
        self._validation_report = self._build_validation_report(df)
        self.processed_data = df.sort_values("Date").reset_index(drop=True)

        print(f"\n✅ Pipeline complete  |  {len(self.processed_data):,} records")
        print(f"   Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
        print(f"   Mandis     : {sorted(df['Mandi_Name'].unique().tolist())}")
        print(f"   Crops      : {sorted(df['Crop'].unique().tolist())}")
        print(f"   Validation : {self._validation_report['quality_score']:.1%} quality score")

        return self.processed_data

    @property
    def validation_report(self) -> Dict:
        return self._validation_report

    # ─────────────────────────────────────────────────────────────────────────
    # Synthetic data generator (for testing without a real dataset)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_rich_synthetic_data(
        self,
        n_days: int = 730,      # 2 years — enough for annual seasonality
        mandis: Optional[List[str]] = None,
        crops: Optional[List[str]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a statistically realistic price panel for testing.

        Includes:
        - Additive trend + dual seasonality (Rabi/Kharif) + noise
        - Crop-specific price levels and volatility
        - Mandi-specific distance premiums
        - ~5 % random missing dates per series (tests imputation)
        - Price shocks (±3 σ events, ~3 per year)
        """
        rng = np.random.default_rng(seed)

        if mandis is None:
            mandis = list(MANDI_BASE_CONFIG.keys())[:8]
        if crops is None:
            crops = TARGET_CROPS

        # Base prices (₹/kg) and parameters per crop
        crop_params = {
            "Onion":  {"base": 18.0, "vol": 4.0, "trend": 0.005},
            "Tomato": {"base": 22.0, "vol": 6.0, "trend": 0.003},
            "Potato": {"base": 14.0, "vol": 2.5, "trend": 0.004},
            "Wheat":  {"base": 20.0, "vol": 1.5, "trend": 0.006},
            "Cotton": {"base": 55.0, "vol": 5.0, "trend": 0.008},
        }

        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")
        records = []

        for mandi in mandis:
            dist_km  = MANDI_BASE_CONFIG.get(mandi, {}).get("distance_km", 150)
            district = MANDI_BASE_CONFIG.get(mandi, {}).get("district", "Unknown")
            dist_premium = dist_km * 0.02  # ₹/kg premium for farther mandis

            for crop in crops:
                p = crop_params.get(crop, {"base": 20.0, "vol": 3.0, "trend": 0.004})
                t = np.arange(n_days)

                trend     = p["base"] + p["trend"] * t + dist_premium
                kharif    = p["vol"] * 0.6 * np.sin(2 * np.pi * t / 182.5)
                rabi      = p["vol"] * 0.4 * np.sin(2 * np.pi * t / 182.5 + np.pi)
                noise     = rng.normal(0, p["vol"] * 0.25, n_days)
                prices    = trend + kharif + rabi + noise
                prices    = np.maximum(prices, p["base"] * 0.4)

                # Inject ~3 price shocks per year
                n_shocks = int(n_days / 365 * 3)
                shock_idx = rng.integers(0, n_days, size=n_shocks)
                prices[shock_idx] += rng.choice([-1, 1], n_shocks) * p["vol"] * 2.5

                # Inject ~5 % missing dates
                keep = rng.random(n_days) > 0.05
                for i, (d, px) in enumerate(zip(dates, prices)):
                    if not keep[i]:
                        continue
                    records.append({
                        "Date":       d,
                        "Mandi_Name": mandi,
                        "Crop":       crop,
                        "District":   district,
                        "Price_per_kg": round(float(px), 4),
                        "Raw_Price":  round(float(px) * 100, 2),
                    })

        df = pd.DataFrame(records)
        df = self._add_features(df)
        self.processed_data = df.sort_values("Date").reset_index(drop=True)

        print(f"✅ Synthetic dataset: {len(df):,} records  |  "
              f"{len(mandis)} mandis × {len(crops)} crops × ~{int(n_days*0.95)} days")
        return self.processed_data

    def get_summary_stats(self) -> Dict:
        """Return a human-readable summary of the processed data."""
        if self.processed_data is None:
            return {}
        df = self.processed_data
        return {
            "total_records": len(df),
            "date_range":    {"start": str(df["Date"].min().date()),
                              "end":   str(df["Date"].max().date()),
                              "days":  (df["Date"].max() - df["Date"].min()).days},
            "mandis":        sorted(df["Mandi_Name"].unique().tolist()),
            "crops":         sorted(df["Crop"].unique().tolist()),
            "price_stats":   (df.groupby("Crop")["Price_per_kg"]
                               .agg(["min", "max", "mean", "std"])
                               .round(2).to_dict()),
            "records_per_mandi_crop": (df.groupby(["Mandi_Name", "Crop"])
                                         .size().to_dict()),
            "validation":    self._validation_report,
        }

    def save_processed_data(self, output_path: str) -> None:
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            logger.info("Saved processed data → %s", output_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_mandi_name(raw: str) -> Optional[str]:
        """Strip suffixes like '(Veg.Sub Yard)', clean whitespace."""
        if pd.isna(raw):
            return None
        s = str(raw).strip()
        if "(" in s:
            s = s.split("(")[0].strip()
        return s if s else None

    @staticmethod
    def _standardise_crop(raw: str, target_crops: List[str]) -> Optional[str]:
        """Map commodity variants → canonical crop name."""
        if pd.isna(raw):
            return None
        raw_l = raw.lower()
        for crop in target_crops:
            if crop.lower() in raw_l:
                return crop
        return None

    @staticmethod
    def _winsorise_prices(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Per-crop IQR-based winsorisation (1.5 × IQR rule)."""
        frames = []
        for crop, grp in df.groupby("Crop"):
            q1, q3 = grp[price_col].quantile([0.25, 0.75])
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            lo     = max(lo, 10.0)   # hard floor ₹10/quintal
            grp    = grp[(grp[price_col] >= lo) & (grp[price_col] <= hi)]
            frames.append(grp)
        return pd.concat(frames, ignore_index=True)

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add distance, traffic, seasonality, and calendar features."""
        # Distance from Gandhinagar
        df["Distance_km"] = df.apply(
            lambda r: self._resolve_distance(r.get("Mandi_Name", ""), r.get("District", "")),
            axis=1,
        )
        # Deterministic traffic score (no random state — uses date+mandi hash)
        df["Traffic_Congestion_Score"] = df.apply(
            lambda r: self._traffic_score(r["Distance_km"], r["Date"]), axis=1
        )
        # Seasonality flags
        df["Is_Rabi_Season"]   = df["Date"].dt.month.isin([11, 12, 1, 2, 3]).astype(int)
        df["Is_Kharif_Season"] = df["Date"].dt.month.isin([6, 7, 8, 9, 10]).astype(int)
        df["Is_Weekend"]       = (df["Date"].dt.dayofweek >= 5).astype(int)
        return df

    def _resolve_distance(self, mandi: str, district: str) -> float:
        """Resolve distance: config lookup → district lookup → default."""
        if mandi in MANDI_BASE_CONFIG:
            return MANDI_BASE_CONFIG[mandi]["distance_km"]
        for key, val in MANDI_BASE_CONFIG.items():
            if key.lower() in mandi.lower():
                return val["distance_km"]
        dist_clean = str(district).strip()
        if dist_clean in DISTRICT_DISTANCES:
            return DISTRICT_DISTANCES[dist_clean]
        for key, km in DISTRICT_DISTANCES.items():
            if key.lower() in dist_clean.lower():
                return km
        return 150.0  # conservative default

    @staticmethod
    def _traffic_score(distance_km: float, date: pd.Timestamp) -> float:
        """
        Deterministic traffic score in [0, 1].
        Based on distance band + weekday/weekend.
        No random state — safe for reproducible pipelines.
        """
        base = 0.65 if distance_km < 50 else (0.40 if distance_km < 150 else 0.20)
        weekday_mult = 1.2 if date.dayofweek < 5 else 0.75
        # Small deterministic jitter from date hash (not random)
        h = int(hashlib.md5(str(date.date()).encode()).hexdigest(), 16)
        jitter = ((h % 100) / 100.0 - 0.5) * 0.10
        return float(np.clip(base * weekday_mult + jitter, 0.05, 0.95))

    @staticmethod
    def _build_validation_report(df: pd.DataFrame) -> Dict:
        """Produce a quality score and per-field null counts."""
        total = len(df)
        null_counts = df.isnull().sum().to_dict()
        completeness = 1.0 - sum(null_counts.values()) / (total * len(df.columns))
        per_mc = df.groupby(["Mandi_Name", "Crop"]).size()
        trainable = int((per_mc >= 30).sum())
        return {
            "total_records":        total,
            "null_counts":          null_counts,
            "completeness":         round(completeness, 4),
            "trainable_pairs":      trainable,
            "total_pairs":          len(per_mc),
            "quality_score":        round(completeness * (trainable / max(len(per_mc), 1)), 4),
        }
