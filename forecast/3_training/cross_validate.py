"""
cross_validate.py
=================
Leave-seasons-out cross-validation: are the results an accident of one train/test split?

The live configuration (PhysNet, weather forecast, NOAA 7-day maximum alert) is retrained six times
on the raw NOAA data (cube_raw.npz, seasons 2016-2026). Each time one block of seasons is held out
and the model is trained on all the others, then scored only on the held-out block:

    2016-17 | 2018-19 | 2020-21 | 2022-23 | 2024-25 | 2026

There is no validation season inside a fold, so nothing is tuned: every fold trains for a fixed 20
epochs (the original runs peaked at epochs 17-26; the one-cycle schedule ends at a low learning rate)
and keeps the last epoch. One seed per fold. Physics persistence is used for leads 1-4, as live.

    python forecast/3_training/cross_validate.py        # resumable: trained folds are skipped
    -> 6_saved_models/cv/fold_<seasons>.pt, 5_results/v2_robustness/cross_validation.json
"""

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "4_evaluation"), str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

import evaluate_v2 as ev  # noqa: E402
from dataset import HORIZON, Cube, is_scored, season_of  # noqa: E402
from evaluate_extended import OUT, physics_max7  # noqa: E402
from physnet import LEADS, Config, Drivers, PhysNet  # noqa: E402

from paths import CUBE_RAW, MODELS, WEATHER_EXT  # noqa: E402

SEASONS = range(2016, 2027)
FOLDS = [(2026,), (2024, 2025), (2016, 2017), (2020, 2021), (2018, 2019), (2022, 2023)]   # most informative first
EPOCHS = 20
PHYSICS_LEADS = (1, 2, 3, 4)
CV_DIR = MODELS / "cv"


def samples(cube, seasons):
    """Issue days whose 7-day-lead target is a scored day of `seasons`, with the 7 input days present."""
    out = []
    for i, day in enumerate(cube.dates):
        target = day + dt.timedelta(days=HORIZON)
        if is_scored(target) and season_of(target) in seasons and target in cube.index \
                and all(day - dt.timedelta(days=k) in cube.index for k in range(7)):
            out.append(i)
    return out


def score_fold(cube, model, held):
    idx = [t for t in range(len(cube.dates))
           if any(is_scored(cube.dates[t] + dt.timedelta(days=k)) and season_of(cube.dates[t] + dt.timedelta(days=k)) in held
                  for k in (1, LEADS))
           and all(cube.dates[t] - dt.timedelta(days=j) in cube.index for j in range(7))]
    names = ["Persistence", "Physics persistence", "PhysNet (this fold)", "Hybrid (live setup)"]
    cms = {s: {n: {k: [] for k in range(1, LEADS + 1)} for n in names} for s in held}
    valid, b = cube.valid, model.builder
    for s0 in range(0, len(idx), 16):
        chunk = idx[s0:s0 + 16]
        P = model.predict_all(chunk)[0]
        for i, t in enumerate(chunk):
            for k in range(1, LEADS + 1):
                day = cube.dates[t] + dt.timedelta(days=k)
                j = cube.index.get(day)
                if j is None or not is_scored(day) or season_of(day) not in held:
                    continue
                y = cube.baa7[j][valid]
                net = P[i, k - 1].astype(np.float32).argmax(0)[valid]
                phys = physics_max7(cube, b, t, k)[valid]
                for n, pred in (("Persistence", cube.baa7[t][valid]), ("Physics persistence", phys), ("PhysNet (this fold)", net),
                                ("Hybrid (live setup)", phys if k in PHYSICS_LEADS else net)):
                    cms[season_of(day)][n][k].append(ev.day_cm(y, pred))
    return cms


def main():
    CV_DIR.mkdir(parents=True, exist_ok=True)
    cube = Cube(CUBE_RAW)
    drivers = Drivers(WEATHER_EXT, cube.lats, cube.lons)
    res_file = OUT / "cross_validation.json"
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads(res_file.read_text()) if res_file.exists() else {"folds": {}}
    all_cms = {}
    for held in FOLDS:
        tag = "-".join(map(str, held))
        path = CV_DIR / f"fold_{tag}.pt"
        train = samples(cube, [s for s in SEASONS if s not in held])
        cfg = Config(target="max7", epochs=EPOCHS, seed=0, name=f"PhysNet max7, held out {tag}")
        if path.exists():
            model = PhysNet.load(path, cube, drivers)
            print(f"== fold {tag}: already trained", flush=True)
        else:
            print(f"\n== fold {tag}: training on {len(train)} issue days ({[s for s in SEASONS if s not in held]})", flush=True)
            t0 = time.time()
            model = PhysNet(cfg).fit(cube, drivers, train, [], None)
            model.save(path)
            res["folds"].setdefault(tag, {})["minutes"] = round((time.time() - t0) / 60, 1)
            res["folds"][tag]["history"] = model.history
        cms = score_fold(cube, model, held)
        for s, v in cms.items():
            all_cms[s] = v
            res["folds"].setdefault(tag, {})[str(s)] = {n: {str(k): {m: round(float(x), 4) for m, x in
                                                                    ev.metrics_from_cm(np.sum(c, 0)).items()}
                                                           for k, c in by.items() if c}
                                                       for n, by in v.items()}
            r = res["folds"][tag][str(s)]
            print(f"   season {s}: 7-day accuracy {r['Hybrid (live setup)']['7']['accuracy']:.3f} "
                  f"(persistence {r['Persistence']['7']['accuracy']:.3f}), 14-day {r['Hybrid (live setup)']['14']['accuracy']:.3f} "
                  f"(persistence {r['Persistence']['14']['accuracy']:.3f})", flush=True)
        # pooled over every held-out season scored so far
        pooled = {}
        for n in ("Persistence", "Physics persistence", "Hybrid (live setup)"):
            pooled[n] = {}
            for k in (1, 3, 7, 14):
                cm = sum(np.sum(all_cms[s][n][k], 0) for s in all_cms if all_cms[s][n][k])
                per = [ev.metrics_from_cm(np.sum(all_cms[s][n][k], 0))["accuracy"] for s in all_cms if all_cms[s][n][k]]
                pooled[n][str(k)] = {**{m: round(float(x), 4) for m, x in ev.metrics_from_cm(cm).items()},
                                     "accuracy_mean_over_seasons": round(float(np.mean(per)), 4),
                                     "accuracy_sd_over_seasons": round(float(np.std(per)), 4),
                                     "seasons": len(per)}
        res["pooled"] = pooled
        res["seasons_scored"] = sorted(int(s) for s in all_cms)
        res_file.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
