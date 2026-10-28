"""
download_raw_cube.py
====================
Builds the forecasting dataset straight from NOAA's raw values instead of the PNG images:
SST anomaly, HotSpot, DHW and SST, daily, 0.1 deg grid (every second 5 km pixel, the same
grid as cube.npz and the live system), from PacIOOS ERDDAP.

    python forecast/1_data_preparation/download_raw_cube.py                # 2015-09 .. latest NOAA day
    python forecast/1_data_preparation/download_raw_cube.py --build-only   # rebuild cube_raw.npz from the cache
    python forecast/1_data_preparation/download_raw_cube.py --reverse --no-build   # extra worker, newest month first

Resumable: one file per month in forecast/data/raw_days/ (float32, no rounding).
Output: forecast/data/cube_raw.npz with the same layout as cube.npz. The Marine Park mask
(`valid`) is copied from cube.npz so trained models and the live system see the same 1,927 cells.

Compared with the image-decoded cube this has no colour-scale clipping (DHW > 12,
HotSpot > 6, |SST anomaly| > 3 degC were clipped in the PNGs) and no rebuilt 2023 days.
"""

import argparse
import calendar
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]
from coralwatch import noaa  # noqa: E402
from paths import CUBE, CUBE_RAW, RAW_DAYS  # noqa: E402

START = dt.date(2015, 9, 1)     # 2016 and 2017 seasons: checked against the aerial bleaching surveys
VARIABLES = ("ssta", "hotspot", "dhw", "sst")    # order used by cube.npz


def months(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        s = dt.date(y, m, 1)
        e = min(end, dt.date(y, m, calendar.monthrange(y, m)[1]))
        yield s, e
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def download(end, start=START, reverse=False):
    RAW_DAYS.mkdir(parents=True, exist_ok=True)
    todo = list(months(start, end))
    if reverse:
        todo.reverse()
    for n, (s, e) in enumerate(todo, 1):
        f = RAW_DAYS / f"{s:%Y-%m}.npz"
        busy = f.with_suffix(".busy")          # another worker is fetching this month
        if (f.exists() and (e.month != end.month or e.year != end.year)) or (busy.exists() and time.time() - busy.stat().st_mtime < 1800):
            continue                      # finished month (the current month is always refreshed)
        busy.touch()
        for attempt in range(10):
            try:
                dates, lats, lons, x = noaa._pacioos_model_days(s, e, timeout=600, variables=VARIABLES)
                np.savez(f, x=x.astype(np.float32), dates=np.array([d.isoformat() for d in dates]), lats=lats, lons=lons)
                print(f"  {s:%Y-%m}: {len(dates)} days ({n}/{len(todo)})", flush=True)
                break
            except Exception as err:
                wait = min(600, 30 * (attempt + 1))
                print(f"  {s:%Y-%m}: {err} - retrying in {wait}s", flush=True)
                time.sleep(wait)
        else:
            busy.unlink(missing_ok=True)
            raise SystemExit(f"PacIOOS unavailable for {s:%Y-%m} - run again later to resume")
        busy.unlink(missing_ok=True)
        time.sleep(2)


def build():
    ref = np.load(CUBE)
    xs, dates = [], []
    for f in sorted(RAW_DAYS.glob("*.npz")):
        z = np.load(f)
        assert np.allclose(z["lats"], ref["lats"]) and np.allclose(z["lons"], ref["lons"]), f"grid mismatch in {f.name}"
        xs.append(z["x"]); dates += list(z["dates"])
    x = np.concatenate(xs)
    order = np.argsort(dates)
    x, dates = x[order], np.array(dates)[order]
    valid = ref["valid"]
    x[:, :, ~valid] = np.nan
    np.savez_compressed(CUBE_RAW, x=x.astype(np.float32), dates=dates, lats=ref["lats"], lons=ref["lons"], valid=valid)
    nan_share = np.isnan(x[:, :, valid]).mean()
    print(f"Saved {CUBE_RAW}: {len(dates)} days ({dates[0]} .. {dates[-1]}), {valid.sum()} park cells, "
          f"{nan_share:.2%} missing values inside the park", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--no-build", action="store_true", help="only download (for extra parallel workers)")
    ap.add_argument("--start", type=dt.date.fromisoformat, default=START)
    ap.add_argument("--end", type=dt.date.fromisoformat, default=None)
    ap.add_argument("--reverse", action="store_true", help="newest month first (a second worker)")
    args = ap.parse_args()
    if not args.build_only:
        download(args.end or noaa.latest_model_date(), args.start, args.reverse)
    if not args.no_build:
        build()
