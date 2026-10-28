"""
Slide charts for Results & Discussion and the test report (PNG, 200 dpi, light background).

    .venv/bin/python testing/make_charts.py

Reads the saved results (forecast/5_results/...) and test reports (testing/reports/...)
and writes testing/charts/*.png.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "forecast" / "5_results" / "v2_max7_final"
DAILY = ROOT / "forecast" / "5_results" / "v2_daily_target"
REPORTS = Path(__file__).resolve().parent / "reports"
OUT = Path(__file__).resolve().parent / "charts"

# Validated categorical slots 1-3 (pass all-pairs CVD checks) + neutral tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def title(ax, main, sub):
    ax.set_title(main, loc="left", fontsize=14, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=INK2, va="bottom")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def lead_curve():
    res = json.loads((FINAL / "results.json").read_text())
    hyb = json.loads((FINAL / "hybrid.json").read_text())["leads"]
    sel = res["selected"]
    leads = sorted(int(k) for k in res["leads"])
    ours = [hyb[str(k)]["test"]["accuracy"][0] if str(k) in hyb and hyb[str(k)]["choice"] == "physics"
            else res["leads"][str(k)][sel]["accuracy"][0] for k in leads]
    lo = [res["leads"][str(k)][sel]["accuracy"][1] for k in leads]
    hi = [res["leads"][str(k)][sel]["accuracy"][2] for k in leads]
    pers = [res["leads"][str(k)]["Persistence"]["accuracy"][0] for k in leads]
    phys = [res["leads"][str(k)]["Physics persistence"]["accuracy"][0] for k in leads]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    net = [i for i, k in enumerate(leads) if not (str(k) in hyb and hyb[str(k)]["choice"] == "physics")]
    ax.fill_between([leads[i] for i in net], [lo[i] * 100 for i in net], [hi[i] * 100 for i in net],
                    color=BLUE, alpha=0.12, linewidth=0)
    ax.axvspan(0.5, net[0] + 0.5, color=GRID, alpha=0.35, linewidth=0)
    ax.text(net[0] / 2 + 0.5, 47, "days 1–4: physics\n(chosen on validation)", ha="center", fontsize=9, color=INK2)
    for ys, c, lab, ls in [(ours, BLUE, "PhysNet (hybrid)", "-"), (phys, AQUA, "Physics persistence", "--"),
                           (pers, ORANGE, '"Same as today"', ":")]:
        ax.plot(leads, [v * 100 for v in ys], color=c, lw=2, ls=ls, marker="o", ms=4.5, label=lab)
        ax.annotate(lab, (leads[-1], ys[-1] * 100), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK2)
    for k in (1, 7, 14):
        i = leads.index(k)
        ax.annotate(f"{ours[i] * 100:.1f}%", (k, ours[i] * 100), xytext=(10 if k == 1 else 0, 2 if k == 1 else 9),
                    textcoords="offset points", ha="left" if k == 1 else "center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlim(0.5, 14.5); ax.set_ylim(45, 101)
    ax.set_xticks(leads); ax.set_xlabel("Days ahead"); ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.31, 0.0), fontsize=10)
    title(ax, "Forecast accuracy by lead time", "NOAA 7-day maximum alert · test seasons 2024–25 · 1,927 reef cells · band = 95% CI of the network")
    save(fig, "1_accuracy_vs_lead.png")


def comparison_7d():
    res = json.loads((FINAL / "results.json").read_text())["leads"]["7"]
    rows = [("\"Same as today\" (persistence)", res["Persistence"]["accuracy"]),
            ("Physics persistence (NOAA formula)", res["Physics persistence"]["accuracy"]),
            ("Trend + NOAA rule", res["Trend + NOAA rule"]["accuracy"]),
            ("PhysNet, trained with forecast error", res["PhysNet 7-day maximum alert, trained with forecast error (seed 0)"]["accuracy"]),
            ("PhysNet, single model", res["PhysNet + weather forecast, 7-day maximum alert (seed 0)"]["accuracy"]),
            ("PhysNet ensemble of 3 (ours)", res["PhysNet + weather forecast, 7-day maximum alert - ensemble of 3"]["accuracy"])]
    rows.sort(key=lambda r: r[1][0])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(rows))
    vals = [r[1][0] * 100 for r in rows]
    colors = [BLUE if r[0].endswith("(ours)") else CONTEXT for r in rows]
    ax.barh(y, vals, color=colors, height=0.62)
    ax.errorbar(vals, y, xerr=[[v - r[1][1] * 100 for v, r in zip(vals, rows)], [r[1][2] * 100 - v for v, r in zip(vals, rows)]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for i, (v, r) in enumerate(zip(vals, rows)):
        ax.text(r[1][2] * 100 + 0.8, i, f"{v:.1f}%", va="center", fontsize=10,
                fontweight="bold" if r[0].endswith("(ours)") else "normal", color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows], color=INK2)
    ax.set_xlim(50, 92); ax.set_xlabel("Accuracy 7 days ahead (%)"); ax.grid(axis="y", visible=False)
    title(ax, "Comparison with existing methods, 7 days ahead", "Test seasons 2024–25 · error bars = 95% CI (moving-block bootstrap)")
    save(fig, "2_method_comparison_7d.png")


def ablation():
    res = json.loads((DAILY / "results.json").read_text())["leads"]["7"]
    rows = [("\"Same as today\"", "Persistence"),
            ("Ocean data only", "PhysNet (ocean only) (seed 0)"),
            ("+ past weather & ENSO/MJO", "PhysNet + observed weather & climate indices (seed 0)"),
            ("+ weather forecast, no air temp.", "PhysNet + weather forecast without air temperature (seed 0)"),
            ("+ weather forecast, no DHW layer", "Same inputs, no DHW formula layer (seed 0)"),
            ("+ weather forecast (full PhysNet)", "PhysNet + weather forecast (seed 0)"),
            ("Full PhysNet, ensemble of 3", "PhysNet + weather forecast - ensemble of 3")]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    vals = [res[k]["accuracy"][0] * 100 for _, k in rows]
    colors = [CONTEXT if i == 0 else (BLUE if i == len(rows) - 1 else "#8fb6e8") for i in range(len(rows))]
    y = list(range(len(rows)))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + 0.5, yy, f"{v:.1f}%", va="center", fontsize=10, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=INK2)
    ax.axvline(vals[0], color=ORANGE, lw=1.2, ls=":")
    ax.set_xlim(55, 85); ax.set_xlabel("Accuracy 7 days ahead (%), daily alert target"); ax.grid(axis="y", visible=False)
    title(ax, "Ablation: what each part adds", "Daily alert level, test seasons 2024–25 · dotted line = \"same as today\"")
    save(fig, "3_ablation.png")


def real_forecasts():
    r = json.loads((FINAL / "real_forecast_check.json").read_text())["results"]
    rows = [("\"Same as today\"", r["Persistence"]["accuracy"], CONTEXT),
            ("PhysNet +\nreal archived forecasts", r["Forecast-weather model + real archived forecasts"]["accuracy"], BLUE),
            ("PhysNet +\nactual weather", r["Forecast-weather model + observed future weather"]["accuracy"], "#8fb6e8")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    vals = [x[1][0] * 100 for x in rows]
    ax.bar(range(3), vals, color=[x[2] for x in rows], width=0.58)
    ax.errorbar(range(3), vals, yerr=[[v - x[1][1] * 100 for v, x in zip(vals, rows)], [x[1][2] * 100 - v for v, x in zip(vals, rows)]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for i, (v, x) in enumerate(zip(vals, rows)):
        ax.text(i, x[1][2] * 100 + 1, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([x[0] for x in rows], color=INK2)
    ax.set_ylim(40, 95); ax.set_ylabel("Accuracy 7 days ahead (%)"); ax.grid(axis="x", visible=False)
    title(ax, "Realistic test: forecasts actually issued at the time", "Peak months Jan–Apr 2024 & 2025 · 233 forecast days · 95% CI")
    save(fig, "4_real_forecast_test.png")


def performance():
    before = json.loads((REPORTS / "performance_before_fixes.json").read_text())
    after = json.loads((REPORTS / "performance.json").read_text())
    panels = [("Map data response (ms)", before["grid_snapshot"]["median_ms"], after["grid_snapshot"]["median_ms"], "{:.1f} ms"),
              ("Throughput, 50 users (req/s)", before["load_50_users"]["throughput_rps"], after["load_50_users"]["throughput_rps"], "{:.0f}/s"),
              ("First forecast (s)", before["forecast_cold"]["ms"] / 1000, after["forecast_cold"]["ms"] / 1000, "{:.1f} s")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (name, b, a, fmt) in zip(axes, panels):
        ax.bar([0, 1], [b, a], color=[ORANGE, BLUE], width=0.55)
        for i, v in enumerate([b, a]):
            ax.text(i, v * 1.02 + max(b, a) * 0.01, fmt.format(v), ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Before", "After"], color=INK2)
        ax.set_ylim(0, max(b, a) * 1.2); ax.set_title(name, fontsize=11, color=INK, loc="left"); ax.grid(axis="x", visible=False)
    fig.suptitle("Performance testing: before and after the fixes it led to", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "5_performance_before_after.png")


if __name__ == "__main__":
    architecture(); lead_curve(); comparison_7d(); ablation(); real_forecasts(); performance(); case_study()


def case_study():
    """The 2024 case-study maps from evaluation, with a title and the alert-level legend added."""
    from matplotlib.patches import Patch
    img = plt.imread(FINAL / "case_2024.png")
    fig = plt.figure(figsize=(12.6, 4.6))
    ax = fig.add_axes([0, 0.1, 1, 0.78]); ax.imshow(img); ax.axis("off")
    fig.text(0.01, 0.95, "2024 mass bleaching: observed vs forecast alert levels", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.9, "NOAA 7-day maximum alert on 3 Mar 2024 (left) and PhysNet ensemble forecasts issued 3, 7 and 14 days earlier",
             fontsize=10, color=INK2)
    names = ["No Stress", "Watch", "Warning", "Alert Level 1", "Alert Level 2"]
    cols = ["#8ec9f2", "#f8d74a", "#fb953a", "#ff5236", "#d4145a"]
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(cols, names)], loc="lower center",
               ncol=5, frameon=False, fontsize=10)
    save(fig, "6_case_study_2024.png")


def architecture():
    """System architecture diagram for the Implementation slide."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 5.2)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5.2)
    ax.grid(False)
    cols = [("DATA", 0.2, [("NOAA Coral Reef Watch", "SST, anomaly, HotSpot, DHW"), ("Open-Meteo weather", "past + 16-day forecast"),
                           ("ENSO / MJO indices", "NOAA CPC, BoM"), ("Reef reports (PDF)", "AIMS, MMP, Reef Authority")]),
            ("PROCESSING", 3.2, [("Image → value recovery", "KD-tree colour inversion"), ("Dataset builder", "season split, 7-day max labels"),
                                 ("Drivers", "weather on 0.1° grid"), ("Report corpus", "3,180 passages, BM25")]),
            ("MODEL", 6.2, [("PhysNet", "CNN → ConvLSTM"), ("NOAA DHW formula layer", "physics built in"),
                            ("Ensemble of 3 + hybrid", "physics for 1–4 days"), ("LLM + fact-checker", "Qwen 2.5 (free, local)")]),
            ("LIVE SYSTEM", 9.2, [("FastAPI backend", "refresh every 3 h"), ("/api/forecast", "1,927 reef cells × 14 days"),
                                  ("/api/explain", "cited, verified briefing"), ("Website", "live map + AI panel")])]
    fills = ["#eef4fc", "#eef4fc", "#e3eefb", "#d6e6fa"]
    for (head, x, boxes), fill in zip(cols, fills):
        ax.text(x + 1.3, 4.95, head, ha="center", fontsize=11, fontweight="bold", color=BLUE)
        for i, (t, sub) in enumerate(boxes):
            y = 3.85 - i * 1.15
            ax.add_patch(FancyBboxPatch((x, y), 2.6, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=fill, ec=BLUE if head == "MODEL" and i < 2 else AXIS, lw=1.4 if head == "MODEL" and i < 2 else 1))
            ax.text(x + 1.3, y + 0.55, t, ha="center", fontsize=10, fontweight="bold", color=INK)
            ax.text(x + 1.3, y + 0.24, sub, ha="center", fontsize=8.5, color=INK2)
    for x in (2.85, 5.85, 8.85):
        ax.annotate("", xy=(x + 0.3, 2.6), xytext=(x, 2.6), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    save(fig, "0_architecture.png")
