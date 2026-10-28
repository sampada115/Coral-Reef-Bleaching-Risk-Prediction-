"""
download_noaa.py
================
Downloads raw NOAA Coral Reef Watch values (not images) for the study area:
northern + central Great Barrier Reef on a 0.1 deg grid.

Only the warm season is downloaded (15 Oct - 30 Apr), when bleaching heat stress can
occur. The forecast is scored on Nov-Apr, so easy winter "No Stress" days don't
inflate the results.

    python forecast/download_noaa.py                     # seasons 2013/14 ... 2025/26
    python forecast/download_noaa.py --first 2016 --last 2026

Resumable: finished files are skipped; failed ones are retried on the next run.
Files: data/raw/<variable>/<YYYY-MM>.nc, one per variable per season month.

Variables: SST anomaly, HotSpot, Degree Heating Weeks. The Bleaching Alert Area is
computed from HotSpot + DHW with NOAA's rule (coralwatch.geo.baa_from), which
reproduces NOAA's published BAA exactly.

Source: NOAA CoastWatch ERDDAP, Coral Reef Watch 5 km daily v3.1 datasets.
"""

import argparse
import calendar
import datetime as dt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]
from coralwatch import noaa  # noqa: E402
from paths import RAW_NOAA as RAW_DIR  # noqa: E402
SEASON_START = (10, 15)  # 15 Oct: two weeks of look-back before November
SEASON_END = (4, 30)


def season_months(first, last):
    """(start_date, end_date) per month for seasons first..last (named by the year they end)."""
    for year in range(first, last + 1):
        for y, m in [(year - 1, 10), (year - 1, 11), (year - 1, 12)] + [(year, k) for k in range(1, 5)]:
            start = dt.date(y, m, SEASON_START[1] if m == 10 else 1)
            end = dt.date(y, m, calendar.monthrange(y, m)[1])
            yield start, end


def fetch(var, start, end, retries=6, pause=3.0):
    out = RAW_DIR / var / f"{start:%Y-%m}.nc"
    if noaa.is_netcdf(out):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            data = noaa.fetch_nc(var, start, end, timeout=180)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(out)
            time.sleep(pause)  # be polite: NOAA's servers block clients that send bursts
            return f"{var} {start:%Y-%m} ok"
        except Exception as e:  # network errors, timeouts, server hiccups
            if attempt == retries:
                return f"{var} {start:%Y-%m} FAILED: {e}"
            time.sleep(60 * attempt)  # a block usually lifts within a few minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2014, help="first season (year it ends in)")
    ap.add_argument("--last", type=int, default=2026, help="last season (year it ends in)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    latest = dt.date.today() - dt.timedelta(days=3)
    jobs = [(v, s, min(e, latest)) for s, e in season_months(args.first, args.last) if s <= latest
            for v in noaa.MODEL_VARIABLES]
    print(f"{len(jobs)} files to check, {args.workers} workers", flush=True)
    failed = done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(fetch, *j) for j in jobs]):
            msg = fut.result()
            done += 1
            if msg:
                failed += "FAILED" in msg
                print(f"[{done}/{len(jobs)}] {msg}", flush=True)
    print("Done." if not failed else f"Done with {failed} failures - run again to retry.", flush=True)


if __name__ == "__main__":
    main()
