"""
download_noaa_outlook.py
========================
NOAA Coral Reef Watch Four-Month Coral Bleaching Outlook (v5, CFSv2-based, 0.5 deg, weekly),
the operational forecast our model is compared with.

Each weekly issue (initial-condition week ending Sunday S) predicts the Bleaching Alert level for
the weeks starting two weeks later. It is published at 10 probability levels: the "060perc" map is
the level reached by at least 60% of the ensemble members (NOAA's headline map), and so on.

For every issue with S between 15 Oct and 30 Apr of the 2024, 2025 and 2026 seasons, this keeps
weeks 1-4 of all 10 probability levels over the study area:
    data/noaa_outlook/icwk<YYYYMMDD>.npz   baa [10 levels, 4 weeks, lat, lon] (int8, -1 = no data),
                                            perc, lats, lons, week_dates (the dates in the file name)

    python forecast/1_data_preparation/download_noaa_outlook.py
"""

import datetime as dt
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]
from paths import NOAA_OUTLOOK as OUT  # noqa: E402

BASE = "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/outlook/v5/nc/v1/outlook"
PERC = list(range(10, 101, 10))
WEEKS = 4
NORTH, SOUTH, WEST, EAST = -10.0, -21.5, 142.0, 153.5
NAME = re.compile(r"cfsv2_outlook-(\d{3})perc_4mon-and-wkly_v5_icwk(\d{8})_for_(\d{8})to(\d{8})\.nc")


def get(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CoralWatch capstone research)"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as err:
            print(f"    {url.rsplit('/', 1)[-1]}: {err} - retry", flush=True)
            time.sleep(15 * (attempt + 1))
    raise SystemExit(f"NOAA STAR unavailable: {url}")


def wanted(day):
    return (day.month == 10 and day.day >= 15) or day.month in (11, 12, 1, 2, 3) or (day.month == 4 and day.day <= 30)


def read_region(data):
    import h5py
    with h5py.File(io.BytesIO(data), "r") as f:
        lat, lon = f["lat"][:], f["lon"][:]
        li = np.flatnonzero((lat <= NORTH) & (lat >= SOUTH))
        lj = np.flatnonzero((lon >= WEST) & (lon <= EAST))
        sl = (slice(li.min(), li.max() + 1), slice(lj.min(), lj.max() + 1))
        weeks = []
        for w in range(1, WEEKS + 1):
            a = f[f"CRW_BAA_Week_{w:02d}"][0][sl].astype(np.int8)
            weeks.append(np.where((a < 0) | (a > 4), -1, a))
        return np.stack(weeks), lat[sl[0]], lon[sl[1]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for year in (2023, 2024, 2025, 2026):
        listing = get(f"{BASE}/{year}/").decode()
        for m in NAME.finditer(listing):
            perc, ic = int(m.group(1)), dt.datetime.strptime(m.group(2), "%Y%m%d").date()
            if dt.date(2023, 10, 15) <= ic <= dt.date(2026, 4, 30) and wanted(ic):
                files.setdefault(ic, {})[perc] = (year, m.group(0), m.group(3))
    issues = sorted(ic for ic, f in files.items() if set(f) >= set(PERC))
    print(f"{len(issues)} outlook issues with all 10 probability levels", flush=True)
    for n, ic in enumerate(issues, 1):
        out = OUT / f"icwk{ic:%Y%m%d}.npz"
        if out.exists():
            continue
        maps = []
        for p in PERC:
            year, name, first = files[ic][p]
            baa, lats, lons = read_region(get(f"{BASE}/{year}/{name}"))
            maps.append(baa)
        first = dt.datetime.strptime(files[ic][60][2], "%Y%m%d").date()
        np.savez_compressed(out, baa=np.stack(maps), perc=np.array(PERC), lats=lats, lons=lons,
                            week_dates=np.array([(first + dt.timedelta(days=7 * w)).isoformat() for w in range(WEEKS)]))
        print(f"  {ic} ({n}/{len(issues)})", flush=True)


if __name__ == "__main__":
    main()
