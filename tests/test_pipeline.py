"""
Run from mandi_intelligence/:  python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_arbitrage.arbitrage_engine import ArbitrageEngine, transport_cost  # noqa: E402
from ml_arbitrage.data_loader import clean_mandi_name, load_prices  # noqa: E402
from ml_arbitrage.distance_calculator import UnknownLocationError, get_distance  # noqa: E402
from ml_arbitrage.features import FEATURE_COLS, build_series_frames  # noqa: E402
from ml_arbitrage.service import dataset_path  # noqa: E402


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return load_prices([dataset_path()])


# ── Data ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("Ahmedabad(Chimanbhai Patal Market Vasana) APMC", "Ahmedabad"),
    ("Rajkot(Veg.Sub Yard)", "Rajkot"),
    ("Nadiyad(Piplag) APMC", "Nadiad"),
    ("Vankaner(Sub yard)", "Wankaner"),
    ("Surat", "Surat"),
])
def test_mandi_names_are_canonical(raw, expected):
    assert clean_mandi_name(raw) == expected


def test_panel_is_one_row_per_mandi_crop_day(panel):
    assert not panel.duplicated(["Mandi_Name", "Crop", "Date"]).any()
    assert panel["Price_per_kg"].between(1, 250).all()


# ── Leakage ──────────────────────────────────────────────────────────────────

def test_features_do_not_use_future_data(panel):
    """Features at origin t must be identical whether or not data after t exists."""
    origin = panel["Date"].min() + pd.Timedelta(days=500)
    full = build_series_frames(panel)
    past = build_series_frames(panel[panel["Date"] <= origin])
    cols = [c for c in FEATURE_COLS if c not in ("horizon", "seasonal_fwd")]
    for key, f_past in past.items():
        a = full[key].loc[origin, cols].astype(float).to_numpy()
        b = f_past.loc[origin, cols].astype(float).to_numpy()
        np.testing.assert_allclose(a, b, equal_nan=True, err_msg=str(key))


# ── Profit engine ────────────────────────────────────────────────────────────

def _forecast_table() -> pd.DataFrame:
    rows = []
    for mandi, base in (("Ahmedabad", 20.0), ("Rajkot", 22.0)):
        for d in range(8):
            p = base + 0.1 * d
            rows.append({"Mandi_Name": mandi, "Crop": "Onion", "Day_Ahead": d,
                         "Predicted_Price": p, "Lower_Bound": p * 0.9 if d else p,
                         "Upper_Bound": p * 1.1 if d else p, "Is_Observed": d == 0})
    return pd.DataFrame(rows)


def test_spoilage_is_counted_once():
    s = ArbitrageEngine._scenario(
        mandi="Rajkot", distance_km=237, qty_quintal=20, wait_days=5, lam=0.03,
        price=20.0, price_lower=18.0, price_upper=22.0, is_forecast=True,
        transport=transport_cost(237, 20),
    )
    sold = 2000 * math.exp(-0.03 * 5)
    assert s["gross_revenue"] == pytest.approx(sold * 20.0, abs=0.01)
    assert s["net_profit"] == pytest.approx(
        s["gross_revenue"] - s["transport_cost"] - s["storage_cost"], abs=0.02)
    # The profit band is in rupees for the kilograms actually sold
    assert s["net_profit"] - s["net_profit_lower"] == pytest.approx(2.0 * sold, abs=0.02)


def test_transport_scales_with_trips():
    one, d1 = transport_cost(100, 20)      # LCV, 1 trip
    two, d2 = transport_cost(100, 400)     # heavy truck, 3 trips
    assert d1["vehicle"] == "LCV" and d1["trips"] == 1
    assert d2["vehicle"] == "HEAVY_TRUCK" and d2["trips"] == 3
    assert transport_cost(0, 20)[0] == 0.0


def test_recommendation_shape_and_alternatives_are_distinct_mandis():
    rec = ArbitrageEngine().get_best_selling_strategy(2000, "Onion", _forecast_table(), "Gandhinagar")
    assert rec["optimal_strategy"]["mandi"] in ("Ahmedabad", "Rajkot")
    names = [s["mandi_name"] for s in rec["alternative_scenarios"]]
    assert len(names) == len(set(names))
    assert rec["optimal_strategy"]["mandi"] not in names


def test_waiting_requires_gain_beyond_uncertainty():
    """A tiny expected rise with a wide interval must not beat selling today."""
    fc = _forecast_table()
    fc = fc[fc["Mandi_Name"] == "Ahmedabad"].copy()
    fc.loc[fc["Day_Ahead"] > 0, "Lower_Bound"] = fc["Predicted_Price"] * 0.7
    rec = ArbitrageEngine().get_best_selling_strategy(500, "Potato", fc, "Ahmedabad")
    assert rec["optimal_strategy"]["days_to_wait"] == 0


# ── Distances ────────────────────────────────────────────────────────────────

def test_distances():
    assert get_distance("Gandhinagar", "Ahmedabad") == 26
    assert get_distance("Ahmedabad", "Gandhinagar") == 26
    assert 150 < get_distance("Bhuj", "Rajkot") < 260
    with pytest.raises(UnknownLocationError):
        get_distance("Atlantis", "Surat")


# ── API contract (what the Nuxt frontend reads) ──────────────────────────────

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.ml_api import app
    with TestClient(app) as c:
        yield c


def test_api_recommend_contract(client):
    r = client.post("/recommend", json={"crop": "onion", "quantity": 2000,
                                        "farmer_location": "Gandhinagar"})
    assert r.status_code == 200, r.text
    data = r.json()
    for key in ("summary", "quantity", "crop", "best_option", "alternatives",
                "optimal_strategy", "justification", "request_id"):
        assert key in data
    best = data["best_option"]
    for key in ("mandi_name", "distance_km", "net_profit", "current_price", "gross_revenue",
                "transport_cost", "storage_cost", "recommendation", "predicted_price_7d"):
        assert key in best
    for alt in data["alternatives"]:
        for key in ("mandi_name", "distance_km", "current_price", "net_profit", "recommendation"):
            assert key in alt
    assert r.headers["X-Request-ID"] == data["request_id"]


def test_api_rejects_unknown_location_and_crop(client):
    assert client.post("/recommend", json={"crop": "Onion", "quantity": 10,
                                           "farmer_location": "Atlantis"}).status_code == 422
    assert client.post("/recommend", json={"crop": "Wheat", "quantity": 10}).status_code == 422


def test_api_health_and_report(client):
    assert client.get("/health").json()["status"] == "healthy"
    report = client.get("/model_report").json()
    assert {r["crop"] for r in report} == {"Onion", "Tomato", "Potato"}
