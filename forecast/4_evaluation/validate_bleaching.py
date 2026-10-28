"""
validate_bleaching.py
=====================
Does the forecast of NOAA's heat-stress alert say anything about bleaching that was actually seen?

Observed bleaching: the aerial surveys of the 2016 and 2017 mass bleaching (Hughes et al. 2016, 2017;
AIMS 2017), compiled by AIMS for NESP TWQ 5.7 (eAtlas, CC BY 3.0 AU). Each surveyed reef has a
category 0-4: 0 = <1% of corals bleached, 1 = 1-10%, 2 = 10-30%, 3 = 30-60%, 4 = >60%.
Both summers come before the model's training data (2018-2021), so this is fully out of sample.

For every surveyed reef inside the study area (nearest Marine Park cell within 0.1 deg) and its
survey day d, scores compared (AUC = chance a bleached reef ranks above an unbleached one):
    NOAA's observed 7-day maximum alert and DHW on day d       (the target itself: the ceiling)
    our forecast issued 7 and 14 days before d                  (chance of Alert Level 1+, and the level)
    "same as today" and physics persistence issued at the same times
for severe bleaching (category >= 3, >30% bleached) and moderate-or-worse (>= 2, >10%).

    python forecast/4_evaluation/validate_bleaching.py
    -> forecast/5_results/v2_robustness/observed_bleaching.json
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

import evaluate_v2 as ev  # noqa: E402
from dataset import Cube  # noqa: E402
from evaluate_extended import OUT, calibrated_alert, load_models, physics_max7  # noqa: E402
from physnet import Drivers  # noqa: E402

from paths import BLEACHING_DB, CUBE_RAW, WEATHER_EXT  # noqa: E402

SURVEYS = BLEACHING_DB / "aerial"
N_BOOT = 1000


def load_surveys():
    rows = []
    for name in ("hughes-aerial-2016", "hughes-aerial-2017", "aims-aerial-2017"):
        for f in json.loads((SURVEYS / f"{name}.json").read_text())["features"]:
            p = {k.lower(): v for k, v in f["properties"].items()}
            raw = p.get("datesurvey") or p.get("surveydate")
            day = (dt.datetime.strptime(raw, "%d/%m/%Y").date() if "/" in raw else dt.date.fromisoformat(raw[:10]))
            rows.append({"source": name, "lat": float(p["latitude"]), "lon": float(p["longitude"]),
                         "category": int(p["bleachcat"]), "date": day})
    return rows


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def auc_ci(score, y, rng):
    vals = []
    for _ in range(N_BOOT):
        s = rng.integers(0, len(y), len(y))
        if 0 < y[s].sum() < len(s):
            vals.append(auc(score[s], y[s]))
    return [round(auc(score, y), 3), round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def main():
    cube = Cube(CUBE_RAW)
    drivers = Drivers(WEATHER_EXT, cube.lats, cube.lons)
    meta, models = load_models(cube, drivers)
    b = models[0].builder
    valid = cube.valid
    cells = np.argwhere(valid)
    clat, clon = cube.lats[cells[:, 0]], cube.lons[cells[:, 1]]

    surveys = []
    for r in load_surveys():
        d2 = (clat - r["lat"]) ** 2 + (clon - r["lon"]) ** 2
        n = int(d2.argmin())
        if d2[n] > 0.1 ** 2 or r["date"] not in cube.index:
            continue
        r["cell"] = tuple(cells[n])
        surveys.append(r)
    print(f"{len(surveys)} surveys inside the study area", flush=True)

    issue_days = sorted({r["date"] - dt.timedelta(days=k) for r in surveys for k in (7, 14)})
    issue_days = [d for d in issue_days if d in cube.index and all(d - dt.timedelta(days=j) in cube.index for j in range(7))]
    idx = [cube.index[d] for d in issue_days]
    probs = {}
    for s in range(0, len(idx), 16):
        chunk = idx[s:s + 16]
        P = np.mean([m.predict_all(chunk)[0].astype(np.float32) for m in models], 0)
        for i, t in enumerate(chunk):
            probs[cube.dates[t]] = P[i]
        print(f"  forecasts {min(s + 16, len(idx))}/{len(idx)}", flush=True)

    cols = {k: [] for k in ("NOAA observed alert on survey day", "NOAA observed DHW on survey day",
                            "Our forecast 7 days before: chance of Alert", "Our forecast 7 days before: level",
                            "Our forecast 14 days before: chance of Alert", "Our forecast 14 days before: level",
                            "Same as today, 7 days before", "Same as today, 14 days before",
                            "Physics persistence, 14 days before")}
    cat, src = [], []
    for r in surveys:
        d, (i, j) = r["date"], r["cell"]
        t7, t14 = d - dt.timedelta(days=7), d - dt.timedelta(days=14)
        if t7 not in probs or t14 not in probs:
            continue
        di = cube.index[d]
        cols["NOAA observed alert on survey day"].append(cube.baa7[di][i, j])
        cols["NOAA observed DHW on survey day"].append(cube.x[di, 2][i, j])
        for k, t in ((7, t7), (14, t14)):
            P = probs[t][k - 1]
            cols[f"Our forecast {k} days before: chance of Alert"].append(calibrated_alert(meta, P[:, i:i + 1, j:j + 1], k)[0, 0])
            cols[f"Our forecast {k} days before: level"].append(P[:, i, j].argmax())
            cols[f"Same as today, {k} days before"].append(cube.baa7[cube.index[t]][i, j])
        cols["Physics persistence, 14 days before"].append(physics_max7(cube, b, cube.index[t14], 14)[i, j])
        cat.append(r["category"]); src.append(r["source"])
    cat = np.array(cat)
    cols = {k: np.nan_to_num(np.array(v, float), nan=-1) for k, v in cols.items()}
    rng = np.random.default_rng(0)
    out = {"surveys_used": int(len(cat)), "by_source": {s: int(src.count(s)) for s in sorted(set(src))},
           "category_counts": {int(c): int((cat == c).sum()) for c in range(5)},
           "note": "AUC: 0.5 = no better than chance, 1 = perfect ranking of bleached above unbleached reefs. "
                   "Bootstrap 95% CIs over surveyed reefs.",
           "auc_severe_bleaching_(>30%)": {k: auc_ci(v, (cat >= 3).astype(int), rng) for k, v in cols.items()},
           "auc_moderate_or_worse_(>10%)": {k: auc_ci(v, (cat >= 2).astype(int), rng) for k, v in cols.items()},
           "share_severe_by_forecast_level_14d": {
               ev_name: {int(l): [round(float((cat[cols["Our forecast 14 days before: level"] == l] >= 3).mean()), 3),
                                  int((cols["Our forecast 14 days before: level"] == l).sum())]
                         for l in range(5) if (cols["Our forecast 14 days before: level"] == l).any()}
               for ev_name in ["share_severe, reefs"]}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "observed_bleaching.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
