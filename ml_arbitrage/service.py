"""
service.py — One object that owns data, models and cached forecasts.

Shared by the FastAPI app and the CLI so both behave identically:

    svc = MandiService.start()          # load data + saved models (train if stale)
    svc.recommend("Onion", 2000, "Gandhinagar")

Forecasts depend only on the data and the model, so they are computed once per
crop and reused; a recommendation is then pure arithmetic over ~60 scenarios.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .arbitrage_engine import ArbitrageEngine
from .data_loader import TARGET_CROPS, active_series, load_prices
from .distance_calculator import is_known_location
from .forecaster import PriceForecaster

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "commodity_price.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def dataset_path() -> Path:
    """Dataset to use; override with the MANDI_DATASET environment variable."""
    return Path(os.environ.get("MANDI_DATASET", DEFAULT_DATASET))


class MandiService:
    def __init__(self, panel: pd.DataFrame, forecaster: PriceForecaster) -> None:
        self.panel = panel
        self.forecaster = forecaster
        self.engine = ArbitrageEngine()
        self.active = active_series(panel)
        self._forecasts: Dict[str, pd.DataFrame] = {}

    @classmethod
    def start(cls, retrain: bool = False, models_dir: Path = MODELS_DIR) -> "MandiService":
        t0 = time.perf_counter()
        path = dataset_path()
        panel = load_prices([path])
        forecaster = PriceForecaster(models_dir)
        fingerprint = {"dataset": path.name, "panel_rows": len(panel),
                       "data_end": str(panel["Date"].max().date())}

        loaded = not retrain and forecaster.load(TARGET_CROPS)
        if loaded and any(m.get("fingerprint") != fingerprint for m in forecaster.meta.values()):
            logger.info("Saved models were trained on different data — retraining.")
            loaded = False
        if not loaded:
            logger.info("Training forecasters (one per crop) …")
            forecaster.fit(panel, TARGET_CROPS)
            for meta in forecaster.meta.values():
                meta["fingerprint"] = fingerprint
            forecaster.save()

        svc = cls(panel, forecaster)
        svc.warm()
        logger.info("Service ready in %.1fs (%s models)", time.perf_counter() - t0,
                    "loaded" if loaded else "trained")
        return svc

    # ── Forecasts ────────────────────────────────────────────────────────────

    def warm(self) -> None:
        for crop in self.crops:
            self.forecast(crop)

    @property
    def crops(self) -> List[str]:
        return [c for c in TARGET_CROPS if c in self.forecaster.models]

    def mandis(self, crop: str) -> List[str]:
        """Active mandis for `crop` whose location we can route to."""
        names = self.active.loc[self.active["Crop"] == crop, "Mandi_Name"].tolist()
        unknown = [m for m in names if not is_known_location(m)]
        if unknown:
            logger.warning("Skipping %s mandis without coordinates (add them to "
                           "distance_calculator._CITY_COORDS): %s", crop, unknown)
        return [m for m in names if m not in unknown]

    def forecast(self, crop: str) -> pd.DataFrame:
        if crop not in self._forecasts:
            self._forecasts[crop] = self.forecaster.forecast(self.panel, crop,
                                                             mandis=self.mandis(crop))
        return self._forecasts[crop]

    # ── Recommendation ───────────────────────────────────────────────────────

    def recommend(self, crop: str, quantity_kg: float, location: str = "Gandhinagar",
                  petrol_price: Optional[float] = None) -> Dict:
        return self.engine.get_best_selling_strategy(
            current_qty_kg=quantity_kg, crop=crop, forecast=self.forecast(crop),
            current_location=location, petrol_price=petrol_price,
        )
