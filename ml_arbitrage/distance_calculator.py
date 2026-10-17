"""
distance_calculator.py — Road distances between Gujarat locations.

Resolution order for get_distance(a, b):
  1. Known road distance from the matrix below (Google Maps / NH figures).
  2. Great-circle distance between known town coordinates × 1.25 road factor
     (calibrated on the matrix: Gujarat road/straight-line ratios are ~1.1–1.3).
  3. Routing through a hub city that has matrix distances to both ends.
Unknown locations raise UnknownLocationError rather than guessing.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Road distance matrix (km) — one-way, sourced from Google Maps / NH distances
# Keys are always in lexicographic order (A, B) where A < B.
# ──────────────────────────────────────────────────────────────────────────────

_RAW_DISTANCES: List[Tuple[str, str, float]] = [
    # ── Gandhinagar spoke ────────────────────────────────────────────────────
    ("Gandhinagar", "Ahmedabad",   26),
    ("Gandhinagar", "Mehsana",     62),
    ("Gandhinagar", "Mansa",       55),
    ("Gandhinagar", "Rajkot",     237),
    ("Gandhinagar", "Surat",      273),
    ("Gandhinagar", "Anand",       98),
    ("Gandhinagar", "Bharuch",    211),
    ("Gandhinagar", "Vadodara",   100),
    ("Gandhinagar", "Bhavnagar",  200),
    ("Gandhinagar", "Jamnagar",   330),
    ("Gandhinagar", "Junagadh",   330),
    ("Gandhinagar", "Amreli",     277),
    ("Gandhinagar", "Sidhpur",    105),
    ("Gandhinagar", "Unjha",       90),
    ("Gandhinagar", "Padra",      110),
    ("Gandhinagar", "Vadhvan",    165),
    ("Gandhinagar", "Wankaner",   220),
    ("Gandhinagar", "Gondal",     265),
    ("Gandhinagar", "Bilimora",   290),
    ("Gandhinagar", "Damnagar",   260),
    # ── Ahmedabad spoke ──────────────────────────────────────────────────────
    ("Ahmedabad",  "Mehsana",      64),
    ("Ahmedabad",  "Rajkot",      216),
    ("Ahmedabad",  "Surat",       263),
    ("Ahmedabad",  "Anand",        89),
    ("Ahmedabad",  "Bharuch",     192),
    ("Ahmedabad",  "Vadodara",    110),
    ("Ahmedabad",  "Bhavnagar",   189),
    ("Ahmedabad",  "Jamnagar",    315),
    ("Ahmedabad",  "Amreli",      255),
    ("Ahmedabad",  "Junagadh",    320),
    ("Ahmedabad",  "Wankaner",    196),
    ("Ahmedabad",  "Gondal",      243),
    ("Ahmedabad",  "Unjha",        72),
    ("Ahmedabad",  "Sidhpur",      82),
    # ── Rajkot spoke ─────────────────────────────────────────────────────────
    ("Rajkot",     "Jamnagar",     92),
    ("Rajkot",     "Bhavnagar",   165),
    ("Rajkot",     "Amreli",      117),
    ("Rajkot",     "Junagadh",    104),
    ("Rajkot",     "Gondal",       37),
    ("Rajkot",     "Wankaner",     45),
    ("Rajkot",     "Surat",       296),
    # ── Surat spoke ──────────────────────────────────────────────────────────
    ("Surat",      "Vadodara",    145),
    ("Surat",      "Bharuch",      66),
    ("Surat",      "Anand",       176),
    ("Surat",      "Bilimora",     51),
    # ── Vadodara / Anand ─────────────────────────────────────────────────────
    ("Vadodara",   "Anand",        46),
    ("Vadodara",   "Bharuch",      66),
    ("Vadodara",   "Mehsana",     148),
    ("Vadodara",   "Padra",        20),
    # ── Misc ─────────────────────────────────────────────────────────────────
    ("Mehsana",    "Unjha",        28),
    ("Mehsana",    "Sidhpur",      45),
    ("Bhavnagar",  "Amreli",       90),
    ("Junagadh",   "Amreli",      170),
    ("Jamnagar",   "Junagadh",   110),
]

# Build bi-directional dict with lower-case keys
_DISTANCE_MATRIX: Dict[Tuple[str, str], float] = {}
for _a, _b, _d in _RAW_DISTANCES:
    _DISTANCE_MATRIX[(_a.lower(), _b.lower())] = float(_d)
    _DISTANCE_MATRIX[(_b.lower(), _a.lower())] = float(_d)

# ──────────────────────────────────────────────────────────────────────────────
# Approximate GPS coordinates (lat, lon) for Haversine fallback
# ──────────────────────────────────────────────────────────────────────────────

_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "gandhinagar": (23.2156, 72.6369),
    "ahmedabad":   (23.0225, 72.5714),
    "mehsana":     (23.6002, 72.4076),
    "unjha":       (23.8008, 72.3906),
    "sidhpur":     (23.9191, 72.3617),
    "mansa":       (23.4265, 72.6671),
    "rajkot":      (22.3039, 70.8022),
    "jamnagar":    (22.4707, 70.0577),
    "junagadh":    (21.5222, 70.4579),
    "gondal":      (21.9603, 70.8010),
    "wankaner":    (22.6096, 70.9449),
    "amreli":      (21.6035, 71.2218),
    "bhavnagar":   (21.7645, 72.1519),
    "surat":       (21.1702, 72.8311),
    "bharuch":     (21.7051, 72.9959),
    "vadodara":    (22.3072, 73.1812),
    "padra":       (22.2337, 73.0861),
    "anand":       (22.5645, 72.9289),
    "bilimora":    (20.7713, 72.9610),
    "damnagar":    (21.6966, 71.5237),
    "vadhvan":     (22.7020, 71.6800),
    # Further APMC towns (approximate town-centre coordinates)
    "kalol":       (23.2460, 72.5000),
    "deesa":       (24.2585, 72.1907),
    "palanpur":    (24.1725, 72.4381),
    "patan":       (23.8493, 72.1266),
    "himatnagar":  (23.5970, 72.9630),
    "modasa":      (23.4627, 73.2985),
    "idar":        (23.8380, 73.0020),
    "visnagar":    (23.6980, 72.5470),
    "vijapur":     (23.5620, 72.7470),
    "kadi":        (23.2990, 72.3320),
    "nadiad":      (22.6939, 72.8616),
    "khambhat":    (22.3181, 72.6189),
    "borsad":      (22.4078, 72.8990),
    "kapadvanj":   (23.0230, 73.0710),
    "godhra":      (22.7788, 73.6143),
    "lunawada":    (23.1280, 73.6100),
    "halol":       (22.5030, 73.4720),
    "dahod":       (22.8350, 74.2550),
    "rajpipla":    (21.8700, 73.5030),
    "ankleshwar":  (21.6264, 73.0152),
    "bardoli":     (21.1230, 73.1120),
    "navsari":     (20.9467, 72.9520),
    "valsad":      (20.5992, 72.9342),
    "vapi":        (20.3893, 72.9106),
    "surendranagar": (22.7276, 71.6480),
    "dhrangadhra": (22.9920, 71.4680),
    "morbi":       (22.8173, 70.8370),
    "jasdan":      (22.0370, 71.2050),
    "jetpur":      (21.7540, 70.6230),
    "dhoraji":     (21.7340, 70.4500),
    "upleta":      (21.7400, 70.2830),
    "porbandar":   (21.6417, 69.6293),
    "veraval":     (20.9070, 70.3680),
    "talala":      (21.0530, 70.5330),
    "keshod":      (21.3000, 70.2500),
    "una":         (20.8230, 71.0390),
    "mahuva":      (21.0900, 71.7600),
    "botad":       (22.1700, 71.6680),
    "palitana":    (21.5250, 71.8250),
    "savarkundla": (21.3370, 71.3060),
    "visavadar":   (21.3380, 70.7470),
    "bhuj":        (23.2420, 69.6669),
    "gandhidham":  (23.0753, 70.1337),
    "dwarka":      (22.2394, 68.9678),
}

# Name aliases — maps variant spellings → canonical name
_ALIASES: Dict[str, str] = {
    "amd":          "ahmedabad",
    "baroda":       "vadodara",
    "mahesana":     "mehsana",
    "mehsana":      "mehsana",
    "saurashtra":   "rajkot",
    "gandhi nagar": "gandhinagar",
    "siddhpur":     "sidhpur",
    "nadiyad":      "nadiad",
    "vankaner":     "wankaner",
    "wadhwan":      "vadhvan",
    "talalagir":    "talala",
}

ROAD_FACTOR = 1.25


class UnknownLocationError(ValueError):
    """Raised when a location is neither in the matrix nor in the coordinate table."""

_HUB_CITIES = ["gandhinagar", "ahmedabad", "vadodara", "rajkot", "surat"]


def _normalise(name: str) -> str:
    """Lower-case, strip, apply alias map."""
    n = name.lower().strip()
    return _ALIASES.get(n, n)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def known_locations() -> List[str]:
    """Every location name get_distance() can resolve (title case)."""
    names = {n for pair in _DISTANCE_MATRIX for n in pair} | set(_CITY_COORDS)
    return sorted(n.title() for n in names)


def is_known_location(name: str) -> bool:
    n = _normalise(name)
    return n in _CITY_COORDS or any(n == a for a, _ in _DISTANCE_MATRIX)


@lru_cache(maxsize=4096)
def get_distance(from_location: str, to_location: str) -> float:
    """Road distance in km between two Gujarat locations (cached)."""
    a, b = _normalise(from_location), _normalise(to_location)
    if a == b:
        return 0.0
    if (a, b) in _DISTANCE_MATRIX:
        return _DISTANCE_MATRIX[(a, b)]
    if a in _CITY_COORDS and b in _CITY_COORDS:
        return round(_haversine_km(*_CITY_COORDS[a], *_CITY_COORDS[b]) * ROAD_FACTOR, 1)
    via_hub = _via_hubs(a, b)
    if via_hub is not None:
        return via_hub
    unknown = [n for n in (from_location, to_location) if not is_known_location(n)]
    raise UnknownLocationError(f"Unknown location(s): {unknown}")


def _via_hubs(a: str, b: str) -> Optional[float]:
    """Shortest a → hub → b route over hubs with matrix distances to both ends."""
    routes = [_DISTANCE_MATRIX[(a, h)] + _DISTANCE_MATRIX[(h, b)]
              for h in _HUB_CITIES
              if (a, h) in _DISTANCE_MATRIX and (h, b) in _DISTANCE_MATRIX]
    return round(min(routes), 1) if routes else None


def calculate_distances_from_location(
    farmer_location: str,
    mandis: List[str],
) -> Dict[str, float]:
    """
    Calculate distances from a farmer's location to each mandi.

    Returns
    -------
    dict {mandi_name: distance_km} sorted by distance ascending.
    """
    distances = {mandi: get_distance(farmer_location, mandi) for mandi in mandis}
    return dict(sorted(distances.items(), key=lambda x: x[1]))


def nearest_mandis(farmer_location: str, all_mandis: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
    """Return the top_n nearest mandis as [(name, km), ...]."""
    dists = calculate_distances_from_location(farmer_location, all_mandis)
    return list(dists.items())[:top_n]
