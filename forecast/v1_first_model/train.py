"""
train.py
========
Trains and evaluates every forecaster, picks the best one on the validation seasons,
reports all models on the held-out test seasons, and saves the chosen model for the
web backend.

    python forecast/train.py            # full run
    python forecast/train.py --quick    # fewer epochs / rows, for a smoke test

Outputs (forecast/results/):
    results.json        all metrics, validation + test, plus the model choice
    RESULTS.md          the same as readable tables
    confusion_test.png  confusion matrix of the chosen model on the test seasons
    event_2024.png      observed vs forecast map for the 2024 event
    training_curve.png  CNN-ConvLSTM validation curve
Model: forecast/models/best.*  +  forecast/models/meta.json
"""

import argparse
import datetime as dt
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

from coralwatch import geo  # noqa: E402
from dataset import CUBE, HORIZON, SEQ_LEN, SPLITS, Cube, build_cube  # noqa: E402
from evaluate import confusion, evaluate  # noqa: E402
from models import (CNNConvLSTM, Climatology, GradientBoosting, Persistence,  # noqa: E402
                    TrendPhysics)

from paths import MODELS_V1 as MODELS, RESULTS_V1 as RESULTS  # noqa: E402
SELECT_BY = "macro_f1"


def run(quick=False):
    if not CUBE.exists():
        build_cube()
    cube = Cube()
    idx = {s: cube.samples(s) for s in SPLITS}
    print({s: len(v) for s, v in idx.items()}, "samples (issue days)")

    candidates = [
        Persistence(), Climatology(), TrendPhysics(),
        GradientBoosting(max_rows=200_000 if quick else 1_500_000),
        CNNConvLSTM(epochs=3 if quick else 40),
    ]
    report = {"task": f"Bleaching Alert level of every Marine Park cell, {HORIZON} days ahead, "
                      f"from the previous {SEQ_LEN} days",
              "grid": f"{len(cube.lats)}x{len(cube.lons)} at 0.1 deg, {int(cube.valid.sum())} park cells",
              "splits": {s: f"seasons ending {min(r)}-{max(r)}, {len(idx[s])} issue days" for s, r in SPLITS.items()},
              "selected_by": f"validation {SELECT_BY}", "models": {}}
    fitted = {}
    for model in candidates:
        print(f"\n== {model.name}", flush=True)
        t0 = time.time()
        model.fit(cube, idx["train"], idx["val"])
        fit_s = time.time() - t0
        res = {"fit_seconds": round(fit_s, 1)}
        for split in ("val", "test"):
            preds = [model.predict(cube, t) for t in idx[split]]
            res[split] = evaluate(cube, idx[split], preds)
            print(f"  {split}: acc {res[split]['accuracy']:.3f}  macro-F1 {res[split]['macro_f1']:.3f}  "
                  f"alert-F1 {res[split]['alert_f1']:.3f}  change-acc {res[split]['change_accuracy']:.3f}", flush=True)
        report["models"][model.name] = res
        fitted[model.name] = model

    learned = [n for n in ("Gradient boosting", "CNN-ConvLSTM") if n in fitted]
    best = max(learned, key=lambda n: report["models"][n]["val"][SELECT_BY])
    report["selected_model"] = best
    base = report["models"]["Persistence"]["test"]
    report["selected_vs_persistence_test"] = {
        k: round(report["models"][best]["test"][k] - base[k], 4)
        for k in ("accuracy", "macro_f1", "alert_f1", "change_accuracy", "zone_alert_mae_pp")
    }
    print(f"\nSelected on validation: {best}")

    save_model(fitted[best], cube)
    if "CNN-ConvLSTM" in fitted:
        report["cnn_history"] = fitted["CNN-ConvLSTM"].history
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(report, indent=2))
    write_markdown(report)
    figures(cube, idx, fitted[best], fitted.get("CNN-ConvLSTM"), fitted["Persistence"])
    print(f"Saved {RESULTS}/results.json and RESULTS.md")


def save_model(model, cube):
    MODELS.mkdir(exist_ok=True)
    meta = {"name": model.name, "trained": dt.date.today().isoformat(), "seq_len": SEQ_LEN,
            "horizon": HORIZON, "lats": cube.lats.tolist(), "lons": cube.lons.tolist(),
            "valid": cube.valid.astype(int).tolist()}
    if isinstance(model, CNNConvLSTM):
        model.save(MODELS / "best.pt")
        meta["file"] = "best.pt"
    else:
        with open(MODELS / "best.pkl", "wb") as f:
            pickle.dump(model, f)
        meta["file"] = "best.pkl"
    (MODELS / "meta.json").write_text(json.dumps(meta))


def write_markdown(r):
    keys = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("alert_precision", "Alert precision"),
            ("alert_recall", "Alert recall"), ("alert_f1", "Alert F1"),
            ("change_accuracy", "Accuracy on changing cells"), ("zone_alert_mae_pp", "Zone % at Alert error (pp)")]
    lines = ["# Forecast results", "", f"**Task:** {r['task']}.", "", f"**Grid:** {r['grid']}.", ""]
    lines += [f"- **{s}:** {d}" for s, d in r["splits"].items()]
    lines += ["", f"Model chosen by {r['selected_by']}: **{r['selected_model']}**. "
              "Test seasons were not used for training or model choice.", ""]
    for split, title in (("test", "Test seasons (held out)"), ("val", "Validation seasons")):
        lines += [f"## {title}", "", "| Model | " + " | ".join(k[1] for k in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for name, res in r["models"].items():
            m = res[split]
            cells = [f"{m[k]:.2f}" if k == "zone_alert_mae_pp" else f"{m[k]:.3f}" for k, _ in keys]
            star = " ⭐" if name == r["selected_model"] else ""
            lines.append(f"| {name}{star} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Share of cells whose level changes within 7 days (test): "
              f"{r['models']['Persistence']['test']['changed_share']:.1%}.", "",
              "Persistence scores 0 on changing cells by definition; a useful forecast must beat it there "
              "without losing much elsewhere."]
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")


def figures(cube, idx, best, cnn, persistence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = ["#8ec9f2", "#f8d74a", "#fb953a", "#ff5236", "#d4145a"]
    cmap = ListedColormap(colors)

    preds = [best.predict(cube, t) for t in idx["test"]]
    cm = confusion(cube, idx["test"], preds)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(5), geo.BAA_NAMES, rotation=30, ha="right"); ax.set_yticks(range(5), geo.BAA_NAMES)
    ax.set_xlabel("Forecast"); ax.set_ylabel("Observed"); ax.set_title(f"{best.name}: test seasons (row-normalised)")
    fig.tight_layout(); fig.savefig(RESULTS / "confusion_test.png", dpi=150); plt.close(fig)

    # A test-set event map: target 3 Mar 2024 (the peak alert day)
    target = dt.date(2024, 3, 3)
    issue = target - dt.timedelta(days=HORIZON)
    if issue in cube.index and cube.index[issue] in idx["test"]:
        t = cube.index[issue]
        panels = [("Observed", cube.target(t)), (f"{best.name} forecast", best.predict(cube, t)),
                  ("Persistence", persistence.predict(cube, t))]
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        ext = [cube.lons[0], cube.lons[-1], cube.lats[-1], cube.lats[0]]
        for ax, (title, arr) in zip(axes, panels):
            ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=-0.5, vmax=4.5, extent=ext)
            ax.set_title(title); ax.set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        fig.suptitle(f"Alert level on {target} forecast on {issue} (7 days ahead)")
        fig.tight_layout(); fig.savefig(RESULTS / "event_2024.png", dpi=150); plt.close(fig)

    if cnn is not None and cnn.history:
        h = cnn.history
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.plot([e["epoch"] for e in h], [e["val_macro_f1"] for e in h], marker="o")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro-F1"); ax.set_title("CNN-ConvLSTM training")
        fig.tight_layout(); fig.savefig(RESULTS / "training_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(**vars(ap.parse_args()))
