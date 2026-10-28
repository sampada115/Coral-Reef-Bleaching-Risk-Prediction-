"""
calibrate.py
============
Temperature scaling of the live ensemble's probabilities (NOAA 7-day maximum alert).

The ensemble was slightly over-confident at high probabilities (forecast 0.85 -> observed
0.73 on the test seasons). Temperature scaling fixes this with one number per lead:
    p_T  ∝  p ** (1 / T)          (T > 1 softens, T < 1 sharpens)
T is fitted on the validation seasons only (minimum log-loss); the most likely level never
changes, so accuracy and F1 are untouched - only the "chance of Alert" becomes more honest.

Also tested, and rejected: an isotonic (monotone) correction of the chance of Alert Level 1+ on top,
fitted on validation. It overfits the two validation seasons and makes the test Brier score worse at
leads 9-14 (14 days: 0.0511 -> 0.0565), so only temperature scaling is used live. Its test scores are
kept in calibration.json ("isotonic_rejected").

    python forecast/4_evaluation/calibrate.py
        -> 5_results/v2_max7_final/calibration.json, reliability_calibrated.png
        -> "calibration_T" added to 6_saved_models/v2/meta.json (used by the live forecast)
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

from dataset import SPLITS, Cube, is_scored, season_of  # noqa: E402
from physnet import LEADS, Drivers, PhysNet  # noqa: E402

from paths import MODELS_V2 as MODELS, RESULTS_FINAL as OUT, WEATHER  # noqa: E402
CHUNK = 24
GRID_T = np.exp(np.linspace(np.log(0.5), np.log(3.0), 61))
BINS = np.linspace(0, 1, 11)


def apply_temperature(probs, T, axis=-2):
    """probs [..., 5, ...] -> recalibrated probabilities (class axis `axis`)."""
    logp = np.log(np.clip(probs, 1e-6, 1)) / T
    logp -= logp.max(axis=axis, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=axis, keepdims=True)


def collect(cube, models, idx, seasons, tgt):
    """Ensemble probabilities and targets at every scored (cell, lead) -> per lead: P [N, 5], y [N]."""
    P = {k: [] for k in range(1, LEADS + 1)}
    Y = {k: [] for k in range(1, LEADS + 1)}
    valid = cube.valid
    for s in range(0, len(idx), CHUNK):
        chunk = idx[s:s + CHUNK]
        probs = np.mean([m.predict_all(chunk)[0].astype(np.float32) for m in models], 0)   # [n, L, 5, H, W]
        for i, t in enumerate(chunk):
            for k in range(1, LEADS + 1):
                day = cube.dates[t] + dt.timedelta(days=k)
                j = cube.index.get(day)
                if j is None or not is_scored(day) or season_of(day) not in seasons:
                    continue
                y = tgt[j][valid]
                ok = ~np.isnan(y)
                P[k].append(probs[i, k - 1][:, valid][:, ok].T.astype(np.float16))
                Y[k].append(y[ok].astype(np.int8))
        print(f"  {min(s + CHUNK, len(idx))}/{len(idx)} issue days", flush=True)
    return {k: (np.concatenate(P[k]).astype(np.float32), np.concatenate(Y[k])) for k in P if P[k]}


def fit_isotonic(p, o, knots=40):
    """Monotone map p -> observed frequency, as (x, y) knots for np.interp."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, o)
    x = np.unique(np.quantile(p, np.linspace(0, 1, knots)))
    x = np.unique(np.concatenate([[0.0], x, [1.0]]))
    return [round(float(v), 4) for v in x], [round(float(v), 4) for v in iso.predict(x)]


def with_alert_prob(P, p_alert):
    """Replace the Alert share of P with p_alert (rescaling the other levels) - for scoring only."""
    q = P.copy()
    a = q[:, 3] + q[:, 4]
    r = np.where(a > 0, p_alert / np.maximum(a, 1e-9), 0)
    q[:, 3] *= r; q[:, 4] *= r
    rest = q[:, :3].sum(1)
    q[:, :3] *= ((1 - p_alert) / np.maximum(rest, 1e-9))[:, None]
    return q


def nll(P, y, T):
    q = apply_temperature(P, T, axis=1)
    return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-9, 1)).mean())


def alert_stats(P, y):
    """Brier score for 'Alert Level 1 or higher' and the reliability table."""
    p = P[:, 3] + P[:, 4]
    o = (y >= 3).astype(np.float32)
    rows = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            rows.append({"bin": f"{a:.1f}-{b:.1f}", "mean_forecast": round(float(p[m].mean()), 3),
                         "observed": round(float(o[m].mean()), 3), "cells": int(m.sum())})
    clim = o.mean()
    brier = float(((p - o) ** 2).mean())
    return {"brier": round(brier, 4), "brier_climatology": round(float(((clim - o) ** 2).mean()), 4),
            "brier_skill": round(1 - brier / float(((clim - o) ** 2).mean()), 4),
            "ece": round(float(sum(r["cells"] * abs(r["mean_forecast"] - r["observed"]) for r in rows) / len(p)), 4),
            "reliability": rows}


def plot(before, after, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="Perfect")
    for st, name, c in ((before, "Ensemble", "#d0573a"), (after, "Calibrated", "#1f77b4")):
        r = st["reliability"]
        ax.plot([x["mean_forecast"] for x in r], [x["observed"] for x in r], "o-", color=c,
                label=f"{name} (Brier {st['brier']:.4f}, ECE {st['ece']:.3f})")
    ax.set_xlabel("Forecast chance of Alert Level 1+ (7 days ahead)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability, test seasons")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    cube = Cube()
    drivers = Drivers(WEATHER, cube.lats, cube.lons)
    meta = json.loads((MODELS / "meta.json").read_text())
    models = [PhysNet.load(MODELS / f"{k}.pt", cube, drivers) for k in meta["ensemble"]]
    tgt = cube.baa7

    print("Validation seasons (fit):", flush=True)
    val = collect(cube, models, cube.samples("val"), SPLITS["val"], tgt)
    temps = {}
    for k, (P, y) in val.items():
        losses = [nll(P, y, T) for T in GRID_T]
        temps[k] = round(float(GRID_T[int(np.argmin(losses))]), 3)
        print(f"  lead {k:2d}: T = {temps[k]:.3f}  (log-loss {min(losses):.4f} vs {nll(P, y, 1.0):.4f} at T=1)", flush=True)
    iso = {}
    for k, (P, y) in val.items():
        q = apply_temperature(P, temps[k], axis=1)
        iso[k] = fit_isotonic(q[:, 3] + q[:, 4], (y >= 3).astype(np.float32))

    print("Test seasons (check):", flush=True)
    test = collect(cube, models, cube.samples("test"), SPLITS["test"], tgt)
    out = {"method": "temperature scaling per lead (minimum log-loss), fitted on validation seasons 2022-2023 only; "
                     "an extra isotonic correction was tested and rejected (worse on test at leads 9-14)",
           "temperature_by_lead": temps, "alert_isotonic_by_lead": iso, "test": {}}
    for k, (P, y) in test.items():
        q = apply_temperature(P, temps[k], axis=1)
        pa = np.interp(q[:, 3] + q[:, 4], *iso[k])
        before, after, iso_after = alert_stats(P, y), alert_stats(q, y), alert_stats(with_alert_prob(q, pa), y)
        out["test"][k] = {"log_loss": [round(nll(P, y, 1.0), 4), round(nll(P, y, temps[k]), 4)],
                          "before": before, "after": after, "isotonic_rejected": iso_after}
        if k == 7:
            plot(before, after, OUT / "reliability_calibrated.png")
        print(f"  lead {k:2d}: Brier {before['brier']:.4f} -> {after['brier']:.4f} (isotonic {iso_after['brier']:.4f})   "
              f"ECE {before['ece']:.4f} -> {after['ece']:.4f} (isotonic {iso_after['ece']:.4f})", flush=True)
    (OUT / "calibration.json").write_text(json.dumps(out, indent=1))
    meta["calibration_T"] = [temps[k] for k in range(1, LEADS + 1)]
    (MODELS / "meta.json").write_text(json.dumps(meta))
    print("Saved calibration.json and updated meta.json", flush=True)


if __name__ == "__main__":
    main()
