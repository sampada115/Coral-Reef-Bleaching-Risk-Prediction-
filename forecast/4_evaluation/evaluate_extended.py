"""
evaluate_extended.py
====================
Robustness checks of the live model (PhysNet 7-day-maximum ensemble, trained on 2018-2021 from the
image-decoded data, unchanged here):

  data     raw NOAA values vs the values decoded from the PNG images (2018-2025)
  raw      the model re-scored on raw NOAA data: test seasons 2024 and 2025 (decoded vs raw, same days)
           and the new, never-seen 2026 season (Nov 2025 - Apr 2026); per season and pooled
  ecmwf    real weather forecasts at every lead 1-14: archived ECMWF IFS runs (Sundays, Dec-Apr of the
           2025 and 2026 seasons) instead of the weather that actually happened
  outlook  head-to-head with NOAA Coral Reef Watch's operational Four-Month Coral Bleaching Outlook:
           same target (NOAA 7-day maximum alert for the week ending 14 days after the issue Sunday),
           same reef cells, seasons 2024-2026

    python forecast/4_evaluation/evaluate_extended.py [--part data|raw|ecmwf|outlook]   (default: all)
    -> forecast/5_results/v2_robustness/{data_check,raw_and_2026,real_forecasts_14d,noaa_outlook}.json, RESULTS.md

Needs: cube_raw.npz (download_raw_cube.py), weather_ext.npz (download_weather.py --start 2015-09-01 --end ...),
ecmwf_runs/ (download_ecmwf_runs.py), noaa_outlook/ (download_noaa_outlook.py).
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

import evaluate_v2 as ev  # noqa: E402
from dataset import Cube, is_scored, season_of  # noqa: E402
from physnet import LEADS, Drivers, PhysNet  # noqa: E402

from paths import (CUBE, CUBE_RAW, ECMWF_RUNS, MODELS_V2 as MODELS, NOAA_OUTLOOK, RESULTS,  # noqa: E402
                   WEATHER, WEATHER_EXT)

OUT = RESULTS / "v2_robustness"
SCALE_SHIFT = np.array([20, 50, 20, 27], np.float32)
SCALE_DIV = np.array([10, 40, 8, 3], np.float32)
CHUNK = 16
SHOW_LEADS = (1, 3, 5, 7, 10, 14)


# ── shared helpers ────────────────────────────────────────────────────
def load_models(cube, drivers):
    meta = json.loads((MODELS / "meta.json").read_text())
    return meta, [PhysNet.load(MODELS / f"{k}.pt", cube, drivers) for k in meta["ensemble"]]


def calibrated_alert(meta, probs, k):
    """Chance of Alert Level 1+ at lead k from ensemble probabilities [5, ...], calibrated as in the live system."""
    T = (meta.get("calibration_T") or [1.0] * LEADS)[k - 1]
    logp = np.log(np.clip(probs, 1e-6, 1)) / T
    e = np.exp(logp - logp.max(0, keepdims=True))
    q = e / e.sum(0, keepdims=True)
    return q[3] + q[4]


def physics_max7(cube, builder, t, k):
    return ev.window_max(cube, t, k, lambda q: ev.physics_persistence(builder, cube, t, q))


def summarize(cms, block):
    """{model: {lead: [days, 5, 5]}} -> {model: {lead: metrics with 95% CIs}}"""
    ev.BLOCK = block
    out = {}
    for name, by_lead in cms.items():
        out[name] = {}
        for k, v in by_lead.items():
            if len(v):
                out[name][str(k)] = {m: [round(x, 4) for x in vals] for m, vals in ev.boot(np.array(v))[0].items()}
                out[name][str(k)]["days"] = len(v)
    return out


def pvalues(cms, ref, block, lead):
    ev.BLOCK = block
    return {n: {m: ev.paired_p(np.array(v[lead]), np.array(cms[ref][lead]), m) for m in ("accuracy", "macro_f1", "alert_f1")}
            for n, v in cms.items() if n != ref and len(v.get(lead, []))}


def brier(pairs):
    p = np.concatenate([a for a, _ in pairs]); o = np.concatenate([b for _, b in pairs])
    clim = o.mean()
    b, bc = float(((p - o) ** 2).mean()), float(((clim - o) ** 2).mean())
    return {"brier": round(b, 4), "brier_climatology": round(bc, 4), "brier_skill": round(1 - b / bc, 4), "cells": int(len(p))}


def future_from_run(drivers, path):
    """Archived ECMWF run -> the model's future-weather channels [8, H, W] (week 1 and week 2 means)."""
    z = np.load(path)
    v = z["vals"]                                                   # [15, P, 4], day 1 = issue day + 1
    weeks = []
    for a, b in ((0, 7), (7, 14)):
        pv = (np.nanmean(v[a:b], 0) - SCALE_SHIFT) / SCALE_DIV
        weeks.append((drivers.M @ np.nan_to_num(pv)).T.reshape((4,) + drivers.shape))
    return np.concatenate(weeks).astype(np.float32), float(np.isnan(v[:14]).mean())


# ── data: raw vs decoded ──────────────────────────────────────────────
def part_data():
    """Decoding accuracy (raw clipped to the PNG colour scales vs decoded) and how often raw values
    lie beyond those scales (all days, and scored Nov-Apr days)."""
    import physnet
    from coralwatch import geo
    lo, hi = physnet.INPUT_RANGE
    dec, raw = np.load(CUBE), np.load(CUBE_RAW)
    di = {d: i for i, d in enumerate(dec["dates"])}
    common = [(di[d], i, dt.date.fromisoformat(str(d))) for i, d in enumerate(raw["dates"]) if d in di]
    valid = dec["valid"]
    names = ["ssta", "hotspot", "dhw", "sst"]
    err = {n: [] for n in names}
    beyond = {n: [0, 0] for n in names}; beyond_scored = {n: [0, 0] for n in names}
    agree = []
    for a, b, day in common:
        xd, xr = dec["x"][a][:, valid], raw["x"][b][:, valid]
        for c, n in enumerate(names):
            ok = ~np.isnan(xd[c]) & ~np.isnan(xr[c])
            err[n].append(np.abs(xd[c][ok] - np.clip(xr[c][ok], lo[c], hi[c])).astype(np.float32))
            out = int(((xr[c][ok] < lo[c]) | (xr[c][ok] > hi[c])).sum())
            beyond[n][0] += out; beyond[n][1] += int(ok.sum())
            if is_scored(day):
                beyond_scored[n][0] += out; beyond_scored[n][1] += int(ok.sum())
        bd, br = geo.baa_from(xd[1], xd[2]), geo.baa_from(xr[1], xr[2])
        ok = ~np.isnan(bd) & ~np.isnan(br)
        agree.append((bd[ok] == br[ok]).mean())
    out = {"days_compared": len(common),
           "png_scales": {n: [float(lo[c]), float(hi[c])] for c, n in enumerate(names)},
           "decoding_mean_abs_error": {n: round(float(np.concatenate(e).mean()), 4) for n, e in err.items()},
           "decoding_p99_abs_error": {n: round(float(np.percentile(np.concatenate(e), 99)), 4) for n, e in err.items()},
           "alert_level_agreement": round(float(np.mean(agree)), 4),
           "alert_level_agreement_worst_day": round(float(np.min(agree)), 4),
           "share_beyond_png_scale_all_days": {n: round(v[0] / max(v[1], 1), 4) for n, v in beyond.items()},
           "share_beyond_png_scale_scored_days": {n: round(v[0] / max(v[1], 1), 4) for n, v in beyond_scored.items()},
           "raw_days": len(raw["dates"]), "raw_range": [str(raw["dates"][0]), str(raw["dates"][-1])]}
    (OUT / "data_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    return out


# ── raw: decoded vs raw, and the new 2026 season ──────────────────────
def score_cube(cube, drivers, seasons, label):
    meta, models = load_models(cube, drivers)
    b = models[0].builder
    phys_leads = set(meta.get("hybrid_physics_leads", []))
    idx = [t for t in range(len(cube.dates))
           if any(season_of(cube.dates[t] + dt.timedelta(days=k)) in seasons and is_scored(cube.dates[t] + dt.timedelta(days=k))
                  for k in (1, LEADS))
           and all(cube.dates[t] - dt.timedelta(days=j) in cube.index for j in range(7))]
    names = ["Persistence", "Physics persistence", "Ensemble", "Hybrid (live)"]
    cms = {s: {n: {k: [] for k in range(1, LEADS + 1)} for n in names} for s in list(seasons) + ["all"]}
    probs7 = {s: [] for s in list(seasons) + ["all"]}
    valid = cube.valid
    for s0 in range(0, len(idx), CHUNK):
        chunk = idx[s0:s0 + CHUNK]
        P = np.mean([m.predict_all(chunk)[0].astype(np.float32) for m in models], 0)
        for i, t in enumerate(chunk):
            for k in range(1, LEADS + 1):
                day = cube.dates[t] + dt.timedelta(days=k)
                j = cube.index.get(day)
                if j is None or not is_scored(day) or season_of(day) not in seasons:
                    continue
                y = cube.baa7[j][valid]
                ens = P[i, k - 1].argmax(0)[valid]
                phys = physics_max7(cube, b, t, k)[valid]
                row = {"Persistence": cube.baa7[t][valid], "Physics persistence": phys, "Ensemble": ens,
                       "Hybrid (live)": phys if k in phys_leads else ens}
                for s in (season_of(day), "all"):
                    for n, pred in row.items():
                        cms[s][n][k].append(ev.day_cm(y, pred))
                    if k == 7:
                        ok = ~np.isnan(y)
                        probs7[s].append((calibrated_alert(meta, P[i, 6], 7)[valid][ok], (y[ok] >= 3).astype(np.float32)))
        print(f"  {label}: {min(s0 + CHUNK, len(idx))}/{len(idx)} issue days", flush=True)
    res = {}
    for s in cms:
        res[str(s)] = {"metrics": summarize(cms[s], 7), "p_vs_persistence_7d": pvalues(cms[s], "Persistence", 7, 7),
                       "alert_probability_7d": brier(probs7[s]) if probs7[s] else None}
    return res


def part_raw():
    out = {}
    print("Decoded data (as in the published results), seasons 2024-2025", flush=True)
    c = Cube(CUBE); d = Drivers(WEATHER, c.lats, c.lons)
    out["decoded_2024_2025"] = score_cube(c, d, (2024, 2025), "decoded")
    del c, d
    print("Raw NOAA data, seasons 2024-2026", flush=True)
    import physnet
    c = Cube(CUBE_RAW); d = Drivers(WEATHER_EXT, c.lats, c.lons)
    out["raw_2024_2026"] = score_cube(c, d, (2024, 2025, 2026), "raw")
    # the live system before inputs were clipped to the training ranges (negative HotSpot, SST < 24 degC)
    physnet.CLIP_INPUTS = False
    out["raw_2024_2026_unclipped_inputs"] = score_cube(c, d, (2024, 2025, 2026), "raw, unclipped")
    physnet.CLIP_INPUTS = True
    (OUT / "raw_and_2026.json").write_text(json.dumps(out, indent=1))
    return out


# ── ecmwf: real forecasts, every lead ─────────────────────────────────
def part_ecmwf():
    cube = Cube(CUBE_RAW); drivers = Drivers(WEATHER_EXT, cube.lats, cube.lons)
    meta, models = load_models(cube, drivers)
    b = models[0].builder
    phys_leads = set(meta.get("hybrid_physics_leads", []))
    runs = {dt.date.fromisoformat(p.stem): p for p in sorted(ECMWF_RUNS.glob("*.npz"))}
    idx, future, missing = [], {}, []
    for day, path in runs.items():
        t = cube.index.get(day)
        if t is None or not all(day - dt.timedelta(days=j) in cube.index for j in range(7)):
            continue
        future[t], miss = future_from_run(drivers, path)
        missing.append(miss); idx.append(t)
    names = ["Persistence", "Physics persistence", "Ensemble + weather that happened (ERA5)",
             "Ensemble + real ECMWF forecast", "Hybrid (live) + real ECMWF forecast"]
    cms = {n: {k: [] for k in range(1, LEADS + 1)} for n in names}
    probs = {k: [] for k in (7, 14)}
    valid = cube.valid
    for s0 in range(0, len(idx), CHUNK):
        chunk = idx[s0:s0 + CHUNK]
        P_obs = np.mean([m.predict_all(chunk)[0].astype(np.float32) for m in models], 0)
        P_fc = np.mean([m.predict_all(chunk, future_weather={t: future[t] for t in chunk})[0].astype(np.float32) for m in models], 0)
        for i, t in enumerate(chunk):
            for k in range(1, LEADS + 1):
                day = cube.dates[t] + dt.timedelta(days=k)
                j = cube.index.get(day)
                if j is None or not is_scored(day):
                    continue
                y = cube.baa7[j][valid]
                phys = physics_max7(cube, b, t, k)[valid]
                fc = P_fc[i, k - 1].argmax(0)[valid]
                row = {"Persistence": cube.baa7[t][valid], "Physics persistence": phys,
                       "Ensemble + weather that happened (ERA5)": P_obs[i, k - 1].argmax(0)[valid],
                       "Ensemble + real ECMWF forecast": fc,
                       "Hybrid (live) + real ECMWF forecast": phys if k in phys_leads else fc}
                for n, pred in row.items():
                    cms[n][k].append(ev.day_cm(y, pred))
                if k in probs:
                    ok = ~np.isnan(y)
                    probs[k].append((calibrated_alert(meta, P_fc[i, k - 1], k)[valid][ok], (y[ok] >= 3).astype(np.float32)))
    out = {"issue_days": [cube.dates[t].isoformat() for t in idx], "weather_missing_share": round(float(np.mean(missing)), 4),
           "note": "Issue days are Sundays, 15 Dec - 30 Apr of the 2025 and 2026 seasons; bootstrap blocks of 3 issue weeks.",
           "metrics": summarize(cms, 3),
           "p_vs_persistence": {str(k): pvalues(cms, "Persistence", 3, k) for k in (7, 10, 14)},
           "alert_probability_ecmwf": {str(k): brier(v) for k, v in probs.items() if v}}
    (OUT / "real_forecasts_14d.json").write_text(json.dumps(out, indent=1))
    return out


# ── outlook: NOAA Four-Month Coral Bleaching Outlook ──────────────────
def outlook_maps(cube):
    """{issue Sunday: (level at 60% [H, W], level at 90% [H, W], chance of Alert 1+ [H, W], week-1 date)} on our grid."""
    files = sorted(NOAA_OUTLOOK.glob("icwk*.npz"))
    first = np.load(files[0])
    olat, olon = first["lats"], first["lons"]
    ok_any = np.zeros((len(olat), len(olon)), bool)
    for f in files:
        ok_any |= (np.load(f)["baa"][:, 0] >= 0).all(0)
    oi, oj = np.nonzero(ok_any)
    lat2, lon2 = np.meshgrid(cube.lats, cube.lons, indexing="ij")
    dist = (lat2[..., None] - olat[oi]) ** 2 + (lon2[..., None] - olon[oj]) ** 2
    near = dist.argmin(-1)
    ni, nj = oi[near], oj[near]                                     # nearest outlook pixel with data, per cell
    perc = first["perc"]
    out = {}
    for f in files:
        z = np.load(f)
        w1 = z["baa"][:, 0][:, ni, nj].astype(np.float32)          # [10, H, W]
        w1[w1 < 0] = np.nan
        alert = (w1 >= 3)
        p_alert = np.where(alert.any(0), perc[np.argmax(alert * perc[:, None, None], 0)] / 100.0, 0.0)
        ic = dt.datetime.strptime(f.stem[4:], "%Y%m%d").date()
        out[ic] = (w1[list(perc).index(60)], w1[list(perc).index(90)], p_alert.astype(np.float32),
                   dt.date.fromisoformat(str(z["week_dates"][0])))
    return out


def part_outlook():
    cube = Cube(CUBE_RAW); drivers = Drivers(WEATHER_EXT, cube.lats, cube.lons)
    meta, models = load_models(cube, drivers)
    b = models[0].builder
    maps = outlook_maps(cube)
    runs = {dt.date.fromisoformat(p.stem): p for p in sorted(ECMWF_RUNS.glob("*.npz"))}
    valid = cube.valid
    K = 14
    issues = []
    for ic, (_, _, _, w1) in sorted(maps.items()):
        t = cube.index.get(ic)
        target = ic + dt.timedelta(days=K)
        if t is None or target not in cube.index or not is_scored(target) or season_of(target) not in (2024, 2025, 2026):
            continue
        if not all(ic - dt.timedelta(days=j) in cube.index for j in range(7)):
            continue
        issues.append(t)
    names = ["Persistence", "Physics persistence", "NOAA Outlook (60% map, headline)", "NOAA Outlook (90% map)",
             "Our ensemble + weather that happened (ERA5)", "Our ensemble + real ECMWF forecast"]
    cms = {n: {K: []} for n in names}
    sub = {n: {K: []} for n in names}                       # issues that also have an ECMWF run
    probs = {"NOAA Outlook": [], "Our ensemble (ERA5)": []}
    probs_sub = {"NOAA Outlook": [], "Our ensemble (ECMWF)": []}
    alignment = {"week_ending_S+14": [], "week_ending_S+20": []}
    for s0 in range(0, len(issues), CHUNK):
        chunk = issues[s0:s0 + CHUNK]
        P_obs = np.mean([m.predict_all(chunk)[0][:, K - 1].astype(np.float32) for m in models], 0)
        fut = {t: future_from_run(drivers, runs[cube.dates[t]])[0] for t in chunk if cube.dates[t] in runs}
        P_fc = (np.mean([m.predict_all([t for t in chunk if t in fut], future_weather=fut)[0][:, K - 1].astype(np.float32)
                         for m in models], 0) if fut else None)
        fc_pos = {t: n for n, t in enumerate([t for t in chunk if t in fut])}
        for i, t in enumerate(chunk):
            ic = cube.dates[t]
            m60, m90, pa, w1 = maps[ic]
            y = cube.baa7[cube.index[ic + dt.timedelta(days=K)]][valid]
            row = {"Persistence": cube.baa7[t][valid], "Physics persistence": physics_max7(cube, b, t, K)[valid],
                   "NOAA Outlook (60% map, headline)": m60[valid], "NOAA Outlook (90% map)": m90[valid],
                   "Our ensemble + weather that happened (ERA5)": P_obs[i].argmax(0)[valid]}
            if t in fut:
                row["Our ensemble + real ECMWF forecast"] = P_fc[fc_pos[t]].argmax(0)[valid]
            for n, pred in row.items():
                cms[n][K].append(ev.day_cm(y, pred))
                if t in fut:
                    sub[n][K].append(ev.day_cm(y, pred))
            ok = ~np.isnan(y)
            o = (y[ok] >= 3).astype(np.float32)
            probs["NOAA Outlook"].append((pa[valid][ok], o))
            probs["Our ensemble (ERA5)"].append((calibrated_alert(meta, P_obs[i], K)[valid][ok], o))
            if t in fut:
                probs_sub["NOAA Outlook"].append((pa[valid][ok], o))
                probs_sub["Our ensemble (ECMWF)"].append((calibrated_alert(meta, P_fc[fc_pos[t]], K)[valid][ok], o))
            # which week does NOAA's "week 1" describe? score its 60% map against both readings
            for key, end in (("week_ending_S+14", ic + dt.timedelta(days=14)), ("week_ending_S+20", ic + dt.timedelta(days=20))):
                j = cube.index.get(end)
                if j is not None:
                    alignment[key].append(ev.day_cm(cube.baa7[j][valid], m60[valid]))
        print(f"  outlook: {min(s0 + CHUNK, len(issues))}/{len(issues)} issues", flush=True)
    cms["Our ensemble + real ECMWF forecast"] = {K: sub["Our ensemble + real ECMWF forecast"][K]}
    out = {"lead_days": K, "issues": [cube.dates[t].isoformat() for t in issues],
           "target": "NOAA 7-day maximum alert for the week ending 14 days after the outlook's initial-condition Sunday "
                     "(= the Outlook's first forecast week)",
           "all_issues_2024_2026": summarize({n: v for n, v in cms.items() if n != "Our ensemble + real ECMWF forecast"}, 3),
           "p_vs_noaa_outlook": pvalues({n: v for n, v in cms.items() if n != "Our ensemble + real ECMWF forecast"},
                                        "NOAA Outlook (60% map, headline)", 3, K),
           "issues_with_ecmwf_run": summarize(sub, 3),
           "p_vs_noaa_outlook_ecmwf_issues": pvalues(sub, "NOAA Outlook (60% map, headline)", 3, K),
           "alert_probability": {n: brier(v) for n, v in probs.items()},
           "alert_probability_ecmwf_issues": {n: brier(v) for n, v in probs_sub.items() if v},
           "noaa_week1_alignment_check": {k: {m: round(float(x), 4) for m, x in ev.metrics_from_cm(np.sum(v, 0)).items()}
                                          for k, v in alignment.items() if v}}
    (OUT / "noaa_outlook.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["data", "raw", "ecmwf", "outlook", "all"], default="all")
    part = ap.parse_args().part
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("data", part_data), ("raw", part_raw), ("ecmwf", part_ecmwf), ("outlook", part_outlook)):
        if part in (name, "all"):
            print(f"== {name}", flush=True)
            fn()


if __name__ == "__main__":
    main()
