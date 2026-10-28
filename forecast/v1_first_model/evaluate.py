"""
evaluate.py
===========
Metrics for 7-day-ahead Bleaching Alert forecasts. Every metric is computed over
Marine Park cells on scored (Nov-Apr) target days.

  accuracy          share of cells with the exact level right
  macro_f1          F1 averaged over the 5 levels (rare Alert levels count as much as No Stress)
  alert_*           precision / recall / F1 for "Alert Level 1 or higher" - the decision managers act on
  change_accuracy   accuracy on cells whose level changes over the 7 days
                    (persistence scores 0 here by definition; this is where a forecast earns its keep)
  zone_alert_mae    mean absolute error, in percentage points, of "% of zone area at Alert"
                    for the northern and central zones
"""

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from coralwatch import geo

N_CLASSES = 5


def _stack(cube, idx, preds):
    ys, ps, now = [], [], []
    for t, p in zip(idx, preds):
        m = cube.valid
        ys.append(cube.target(t)[m]); ps.append(p[m]); now.append(cube.baa[t][m])
    y, p, n = (np.concatenate(a) for a in (ys, ps, now))
    ok = ~np.isnan(y) & ~np.isnan(p)
    return y[ok].astype(int), p[ok].astype(int), n[ok].astype(int)


def macro_f1(cube, idx, preds):
    y, p, _ = _stack(cube, idx, preds)
    return float(f1_score(y, p, labels=range(N_CLASSES), average="macro", zero_division=0))


def zone_alert_mae(cube, idx, preds):
    rows = geo.zone_rows(cube.lats)
    errs = []
    for t, p in zip(idx, preds):
        y = cube.target(t)
        for zid, r in rows.items():
            m = cube.valid & r[:, None]
            if m.sum():
                errs.append(abs((y[m] >= 3).mean() - (p[m] >= 3).mean()) * 100)
    return float(np.mean(errs))


def evaluate(cube, idx, preds):
    y, p, now = _stack(cube, idx, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y >= 3, p >= 3, average="binary", zero_division=0)
    changed = y != now
    per_class = precision_recall_fscore_support(y, p, labels=range(N_CLASSES), zero_division=0)
    return {
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, labels=range(N_CLASSES), average="macro", zero_division=0)),
        "alert_precision": float(pr),
        "alert_recall": float(rc),
        "alert_f1": float(f1),
        "change_accuracy": float((y[changed] == p[changed]).mean()) if changed.any() else float("nan"),
        "changed_share": float(changed.mean()),
        "zone_alert_mae_pp": zone_alert_mae(cube, idx, preds),
        "per_class_recall": {geo.BAA_NAMES[k]: float(per_class[1][k]) for k in range(N_CLASSES)},
        "n_cells": int(len(y)),
        "n_days": len(idx),
    }


def confusion(cube, idx, preds):
    y, p, _ = _stack(cube, idx, preds)
    cm = np.zeros((N_CLASSES, N_CLASSES), int)
    np.add.at(cm, (y, p), 1)
    return cm
