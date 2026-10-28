"""
forecast_error.py
=================
Measures how wrong real weather forecasts are, so training can add the same amount of
error to the "future weather" input (the model otherwise learns to trust perfect weather).

Uses archived Open-Meteo forecasts (lead 1-7 days) for May-October 2024 and 2025 - months
the test never scores - compared with ERA5 at the same 74 points.

    python forecast/1_data_preparation/forecast_error.py    # -> forecast/data/forecast_error_stats.json

Stats per variable, in the model's scaled units:
  week1: error of the 7-day mean forecast (leads 1-7)
  week2: error of the lead-7 forecast, a proxy for the unarchived 8-14 day leads
  mean (bias), std, and the share of variance common to all points (large-scale error)
"""

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]
from coralwatch import weather  # noqa: E402
from paths import FORECAST_ERROR_STATS as OUT, PREVRUNS_CACHE as CACHE, WEATHER  # noqa: E402
PERIODS = [(dt.date(2024, 5, 1), dt.date(2024, 10, 31)), (dt.date(2025, 5, 1), dt.date(2025, 10, 31))]
SHIFT = np.array([20, 50, 20, 27], np.float32)
DIV = np.array([10, 40, 8, 3], np.float32)


def main():
    points = weather.weather_points()
    CACHE.mkdir(parents=True, exist_ok=True)
    fc = {j: {} for j in range(1, 8)}
    for s, e in PERIODS:
        for b in range(0, len(points), 8):
            f = CACHE / f"err_{s}_{b:03d}.npz"
            if not f.exists():
                for attempt in range(12):
                    try:
                        r = weather.fetch_previous_runs(points[b:b + 8], s, e)
                        np.savez(f, dates=np.array([d.isoformat() for d in r[1][0]]), **{f"lead{j}": r[j][1] for j in r})
                        break
                    except Exception as err:
                        wait = min(900, 60 * (attempt + 1))
                        print(f"  {s} batch {b}: {err} - retrying in {wait}s", flush=True)
                        time.sleep(wait)
                else:
                    raise SystemExit("Open-Meteo unavailable - run again later")
                time.sleep(3)
            z = np.load(f)
            for j in range(1, 8):
                for d, row in zip(z["dates"], z[f"lead{j}"]):
                    fc[j].setdefault(dt.date.fromisoformat(str(d)), np.full((len(points), 4), np.nan, np.float32))[b:b + 8] = row
        print(f"  archived forecasts {s}..{e} ready", flush=True)

    w = np.load(WEATHER)
    era_idx = {dt.date.fromisoformat(str(d)): i for i, d in enumerate(w["dates"])}
    era = w["vals"]
    e1, e7 = [], []
    for s, e in PERIODS:
        day = s
        while day + dt.timedelta(days=7) <= e:
            errs = []
            for j in range(1, 8):
                d = day + dt.timedelta(days=j)
                if d in fc[j] and d in era_idx:
                    errs.append((fc[j][d] - era[era_idx[d]]) / DIV)
            if len(errs) == 7:
                e1.append(np.nanmean(np.stack(errs), 0))
                e7.append(errs[-1])
            day += dt.timedelta(days=1)
    stats = {}
    for name, arr in (("week1", np.array(e1)), ("week2", np.array(e7))):     # [N, P, 4]
        mean = np.nanmean(arr, (0, 1))
        std = np.nanstd(arr, (0, 1))
        shared = np.nanvar(np.nanmean(arr, 1), 0) / np.maximum(np.nanvar(arr, (0, 1)), 1e-9)
        stats[name] = {"mean": mean.round(4).tolist(), "std": std.round(4).tolist(),
                       "shared": np.clip(shared, 0, 1).round(3).tolist(), "n_issue_days": int(len(arr))}
    stats["variables"] = weather.WEATHER_NAMES
    stats["periods"] = [f"{s}..{e}" for s, e in PERIODS]
    OUT.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
