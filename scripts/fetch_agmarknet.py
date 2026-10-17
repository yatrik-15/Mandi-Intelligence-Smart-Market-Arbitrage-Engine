"""
fetch_agmarknet.py — Download real Gujarat mandi prices from AGMARKNET.

Source: data.gov.in "Variety-wise Daily Market Prices Data of Commodity"
        (Directorate of Marketing & Inspection, Ministry of Agriculture).
        https://www.data.gov.in/resource/variety-wise-daily-market-prices-data-commodity

The public sample API key returns at most 10 records per request and is
throttled to roughly one request every two seconds, so the script walks the
calendar one (crop, date) at a time, newest day first, through a shared rate
limiter. Every finished (crop, date) is appended to a JSONL checkpoint, so the
run can be interrupted and resumed, and later runs only fetch missing days.

Set DATA_GOV_IN_API_KEY to your own (free) key from data.gov.in for higher
limits, and raise --rate accordingly.

Usage
-----
    python scripts/fetch_agmarknet.py                    # 2021-01-01 → today
    python scripts/fetch_agmarknet.py --start 2024-01-01 --workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"
CHECKPOINT = RAW_DIR / "agmarknet_gujarat.jsonl"
OUTPUT_CSV = ROOT / "dataset" / "agmarknet_gujarat.csv"

CROPS = ["Onion", "Tomato", "Potato"]
STATE = "Gujarat"
CSV_COLUMNS = ["Arrival_Date", "State", "District", "Market", "Commodity",
               "Variety", "Grade", "Min_Price", "Max_Price", "Modal_Price"]

# The API silently stalls requests that carry Python's default User-Agent.
HEADERS = {"User-Agent": "BeejRakshak-MandiIntelligence/3.1 (+data.gov.in open data)"}


class RateLimiter:
    """Spaces requests at least `1 / rate` seconds apart across all threads."""

    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate
        self._next = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))

    def penalise(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


_write_lock = threading.Lock()
_limiter = RateLimiter(0.5)


def _get(params: dict, page_size: int, retries: int = 8) -> dict:
    query = {"api-key": os.environ.get("DATA_GOV_IN_API_KEY", SAMPLE_KEY),
             "format": "json", "limit": page_size, **params}
    request = urllib.request.Request(f"{API_URL}?{urllib.parse.urlencode(query)}",
                                     headers=HEADERS)
    for attempt in range(retries):
        _limiter.wait()
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("status") == "ok":
                return payload
            raise RuntimeError(payload.get("message", "bad status"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            _limiter.penalise(10 * (attempt + 1))   # throttled: back off globally
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def fetch_day(crop: str, day: date, page_size: int) -> list[dict]:
    """All Gujarat records for one crop on one date (handles paging)."""
    filters = {"filters[State]": STATE, "filters[Commodity]": crop,
               "filters[Arrival_Date]": day.strftime("%d/%m/%Y")}
    first = _get({**filters, "offset": 0}, page_size)
    records = list(first.get("records", []))
    total = int(first.get("total", 0))
    step = max(len(records), 1)
    for offset in range(step, total, step):
        records.extend(_get({**filters, "offset": offset}, page_size).get("records", []))
    return records


def _done_keys() -> set[tuple[str, str]]:
    if not CHECKPOINT.exists():
        return set()
    with CHECKPOINT.open(encoding="utf-8") as fh:
        return {(row["crop"], row["date"]) for row in map(json.loads, fh)}


def download(start: date, end: date, workers: int, page_size: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = _done_keys()
    # Newest first, so an interrupted run still yields a contiguous recent window.
    days = [end - timedelta(d) for d in range((end - start).days + 1)]
    # The most recent days are re-fetched every run: late reports keep arriving.
    recent = end - timedelta(days=7)
    jobs = [(c, d) for d in days for c in CROPS
            if (c, d.isoformat()) not in done or d >= recent]
    print(f"{len(jobs):,} (crop, date) jobs to fetch, {len(done):,} already cached")

    t0, n_records = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool, \
         CHECKPOINT.open("a", encoding="utf-8") as fh:
        futures = {pool.submit(fetch_day, c, d, page_size): (c, d) for c, d in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            crop, day = futures[fut]
            try:
                records = fut.result()
            except Exception as exc:  # leave the job for the next run
                print(f"  ! {crop} {day}: {exc}", file=sys.stderr)
                continue
            n_records += len(records)
            with _write_lock:
                fh.write(json.dumps({"crop": crop, "date": day.isoformat(),
                                     "records": records}) + "\n")
                fh.flush()
            if i % 60 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                print(f"  {i:,}/{len(jobs):,} jobs  {n_records:,} records  "
                      f"({rate:.1f} jobs/s, ~{(len(jobs) - i) / rate / 60:.0f} min left)")


def export_csv() -> int:
    """Collapse the checkpoint (last write per crop-date wins) into one CSV."""
    latest: dict[tuple[str, str], list[dict]] = {}
    with CHECKPOINT.open(encoding="utf-8") as fh:
        for row in map(json.loads, fh):
            latest[(row["crop"], row["date"])] = row["records"]

    rows = [rec for key in sorted(latest) for rec in latest[key]]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} records → {OUTPUT_CSV.relative_to(ROOT)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2021-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0.5,
                        help="Max requests per second (sample key: ~0.5)")
    parser.add_argument("--page-size", type=int, default=1000,
                        help="Requested page size (the sample key caps it at 10)")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if not args.export_only:
        _limiter.interval = 1.0 / args.rate
        download(datetime.strptime(args.start, "%Y-%m-%d").date(),
                 datetime.strptime(args.end, "%Y-%m-%d").date(),
                 args.workers, args.page_size)
    export_csv()


if __name__ == "__main__":
    main()
