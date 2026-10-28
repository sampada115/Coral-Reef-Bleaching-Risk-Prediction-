"""
download_ecmwf_runs.py
======================
Archived ECMWF IFS forecast runs (Open-Meteo Single Runs API) for the
real-forecast test at every lead 1-14 days: for each issue day, the full 15-day run started that
day at 00 UTC, as daily weather at the 74 points.

Issue days: Sundays (the NOAA Four-Month Outlook's initial-condition dates, so the same days serve the
comparison with NOAA's outlook), 15 Dec - 30 Apr of the 2025 and 2026 seasons (archived runs before ~Oct 2024 stop at 10 days).
With --wednesdays, Wednesdays are added (more forecast days; needs a second day of the free API quota).

    python forecast/1_data_preparation/download_ecmwf_runs.py [--wednesdays]

Resumable (one file per run in data/ecmwf_runs/); waits and retries when the free API's limit is reached.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]
from coralwatch import weather  # noqa: E402
from paths import ECMWF_RUNS as CACHE  # noqa: E402

FIRST, LAST = dt.date(2024, 10, 1), dt.date(2026, 4, 30)   # runs reach 15 days only from about Oct 2024 (10 days before)


def issue_days(weekdays=(6,)):
    d = FIRST
    while d <= LAST:
        if d.weekday() in weekdays and (d.month >= 12 and d.day >= 15 or d.month <= 4):
            yield d
        d += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wednesdays", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    points = weather.weather_points()
    days = list(issue_days((6, 2) if args.wednesdays else (6,)))
    print(f"{len(days)} runs, {len(points)} points", flush=True)
    for n, day in enumerate(days, 1):
        f = CACHE / f"{day.isoformat()}.npz"
        if f.exists():
            continue
        for attempt in range(30):
            try:
                dates, vals = weather.fetch_single_run(points, day)
                np.savez(f, dates=np.array([d.isoformat() for d in dates]), vals=vals)
                print(f"  {day} ok ({n}/{len(days)}, {np.isnan(vals).mean():.1%} missing)", flush=True)
                break
            except Exception as err:
                wait = min(1800, 120 * (attempt + 1))     # free API: calls per minute / hour / day are limited
                print(f"  {day}: {err} - retrying in {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise SystemExit("Open-Meteo unavailable - run again later to resume")
        time.sleep(20)


if __name__ == "__main__":
    main()
