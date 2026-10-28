"""
report_robustness.py
====================
Writes forecast/5_results/v2_robustness/RESULTS.md from the JSON files made by
evaluate_extended.py, calibrate.py, validate_bleaching.py and cross_validate.py
(sections whose JSON doesn't exist yet are skipped).

    python forecast/4_evaluation/report_robustness.py
"""

import json
import sys
from pathlib import Path

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model")]
from paths import RESULTS, RESULTS_FINAL  # noqa: E402

OUT = RESULTS / "v2_robustness"


def load(name, folder=OUT):
    f = folder / name
    return json.loads(f.read_text()) if f.exists() else None


def ci(v):
    return f"{v[0]:.3f} [{v[1]:.3f}–{v[2]:.3f}]" if isinstance(v, list) and len(v) == 3 else f"{v:.3f}"


def table(metrics, leads, keys=("accuracy", "macro_f1", "alert_f1"), names=None):
    rows = ["| Model | Lead | " + " | ".join(k.replace("_", " ").title() for k in keys) + " |",
            "|---|---|" + "---|" * len(keys)]
    for k in leads:
        for n in (names or metrics):
            m = metrics.get(n, {}).get(str(k))
            if m:
                rows.append(f"| {n} | {k} d | " + " | ".join(ci(m[x]) for x in keys) + " |")
    return "\n".join(rows)


def main():
    md = ["# Robustness checks of the live model", "",
          "The live model is unchanged: the PhysNet 7-day-maximum-alert ensemble trained on 2018–2021. "
          "Target everywhere: NOAA's 7-day maximum Bleaching Alert level, 1,927 reef cells. "
          "Brackets are 95% block-bootstrap confidence intervals.", ""]

    d = load("data_check.json")
    if d:
        md += ["## 1. Raw NOAA data instead of values decoded from the PNG images", "",
               f"Compared on {d['days_compared']:,} days (2018–2025), every Marine Park cell.", "",
               "| Variable | PNG colour scale | Decoding error (mean / 99th pct) | Raw values beyond the scale: all days / Nov–Apr |",
               "|---|---|---|---|"]
        md += [f"| {n} | {d['png_scales'][n][0]:g} to {d['png_scales'][n][1]:g} | {d['decoding_mean_abs_error'][n]} / "
               f"{d['decoding_p99_abs_error'][n]} | {d['share_beyond_png_scale_all_days'][n]:.1%} / "
               f"{d['share_beyond_png_scale_scored_days'][n]:.1%} |" for n in d["png_scales"]]
        md += ["", f"Alert level identical on {d['alert_level_agreement']:.1%} of cell-days "
                   f"(worst day {d['alert_level_agreement_worst_day']:.1%}). Values beyond a scale were clipped in the "
                   "images; the model's inputs are now clipped to the same ranges (`physnet.INPUT_RANGE`), live and in "
                   "evaluation, so raw data reach the network exactly as during training.", ""]

    r = load("raw_and_2026.json")
    if r:
        md += ["## 2. Same model, raw data, and the new 2026 season", "",
               "Decoded data = the inputs used for the published results. Raw = NOAA's values straight from PacIOOS ERDDAP, "
               "with ERA5 weather. Season 2026 = Nov 2025 – Apr 2026, never seen in any way.", ""]
        rows = ["| Data | Season | Hybrid (live) 7 d | Persistence 7 d | Hybrid 14 d | Persistence 14 d | Brier skill (chance of Alert, 7 d) |",
                "|---|---|---|---|---|---|---|"]
        for key, label in (("decoded_2024_2025", "Decoded"), ("raw_2024_2026", "Raw"),
                           ("raw_2024_2026_unclipped_inputs", "Raw, unclipped inputs (live system before the fix)")):
            if key not in r:
                continue
            for s, v in r[key].items():
                m = v["metrics"]
                bs = v["alert_probability_7d"]["brier_skill"] if v.get("alert_probability_7d") else float("nan")
                rows.append(f"| {label} | {s} | {ci(m['Hybrid (live)']['7']['accuracy'])} | {ci(m['Persistence']['7']['accuracy'])} | "
                            f"{ci(m['Hybrid (live)']['14']['accuracy'])} | {ci(m['Persistence']['14']['accuracy'])} | {bs:.3f} |")
        md += rows + ["", "All metrics, raw data, all three test seasons pooled:", "",
                      table(r["raw_2024_2026"]["all"]["metrics"], (1, 3, 7, 14)), ""]

    e = load("real_forecasts_14d.json")
    if e:
        md += ["## 3. Real weather forecasts at every lead (archived ECMWF runs)", "",
               f"{len(e['issue_days'])} issue days (Sundays, 15 Dec – 30 Apr of the 2025 and 2026 seasons). "
               "The model is given the ECMWF IFS forecast that was actually issued that day for both forecast weeks, "
               "instead of the weather that happened. Bootstrap blocks of 3 weeks.", "",
               table(e["metrics"], (1, 3, 7, 10, 14)), "",
               "One-sided p-values vs persistence (accuracy): " +
               "; ".join(f"{k} d: " + ", ".join(f"{n} {v['accuracy']:.3f}" for n, v in d_.items() if "ECMWF" in n)
                         for k, d_ in e["p_vs_persistence"].items()), ""]
        if e.get("alert_probability_ecmwf"):
            md += ["Chance of Alert with real forecasts: " + "; ".join(
                f"{k} d Brier skill {v['brier_skill']:.3f}" for k, v in e["alert_probability_ecmwf"].items()), ""]

    o = load("noaa_outlook.json")
    if o:
        md += ["## 4. Against NOAA's operational Four-Month Coral Bleaching Outlook", "",
               f"{len(o['issues'])} weekly Outlook issues (seasons 2024–2026). Target: {o['target']}. "
               "NOAA's 0.5° Outlook is matched to each reef cell by its nearest ocean pixel.", "",
               table(o["all_issues_2024_2026"], (14,), ("accuracy", "macro_f1", "alert_f1", "alert_recall")), "",
               "One-sided p-values that a forecaster is NOT better than NOAA's 60% Outlook: " +
               "; ".join(f"{n}: acc {v['accuracy']:.3f}, alert F1 {v['alert_f1']:.3f}" for n, v in o["p_vs_noaa_outlook"].items()), "",
               "Issues that also have an archived ECMWF run (2025–2026, the fully realistic comparison):", "",
               table(o["issues_with_ecmwf_run"], (14,), ("accuracy", "macro_f1", "alert_f1", "alert_recall")), "",
               "Chance of Alert Level 1+ (Brier skill vs climatology): " +
               "; ".join(f"{n} {v['brier_skill']:.3f}" for n, v in o["alert_probability"].items()) +
               ((" | ECMWF issues: " + "; ".join(f"{n} {v['brier_skill']:.3f}" for n, v in o["alert_probability_ecmwf_issues"].items()))
                if o.get("alert_probability_ecmwf_issues") else ""), "",
               "Check of which week NOAA's 'week 1' describes (its 60% map scored against both readings, accuracy): " +
               "; ".join(f"{k} {v['accuracy']:.3f}" for k, v in o["noaa_week1_alignment_check"].items()), ""]

    b = load("observed_bleaching.json")
    if b:
        md += ["## 5. Against bleaching that was actually seen (2016 and 2017 aerial surveys)", "",
               f"{b['surveys_used']:,} reef surveys ({', '.join(f'{k} {v}' for k, v in b['by_source'].items())}). "
               "Both summers are before the training data. " + b["note"], "",
               "| Score | AUC, severe bleaching (>30%) | AUC, moderate or worse (>10%) |", "|---|---|---|"]
        sev, mod = b["auc_severe_bleaching_(>30%)"], b["auc_moderate_or_worse_(>10%)"]
        md += [f"| {k} | {ci(sev[k])} | {ci(mod[k])} |" for k in sev] + [""]

    c = load("calibration.json", RESULTS_FINAL)
    if c:
        md += ["## 6. Calibrated chance of Alert", "", c["method"] + ".", "",
               "| Lead | T | Brier before | after | ECE before | after | Isotonic (rejected) Brier |", "|---|---|---|---|---|---|---|"]
        for k, v in c["test"].items():
            md.append(f"| {k} d | {c['temperature_by_lead'][k]} | {v['before']['brier']} | {v['after']['brier']} | "
                      f"{v['before']['ece']} | {v['after']['ece']} | {v.get('isotonic_rejected', {}).get('brier', '–')} |")
        md += ["", "Figure: `../v2_max7_final/reliability_calibrated.png`.", ""]

    cv = load("cross_validation.json")
    if cv and cv.get("pooled"):
        md += ["## 7. Leave-seasons-out cross-validation (retrained on raw data)", "",
               "Six folds; each retrains the live configuration on all other seasons (fixed 20 epochs, nothing tuned) "
               "and scores only the held-out seasons. Physics persistence for leads 1–4, as live.", "",
               "| Held-out season | Hybrid 7 d | Persistence 7 d | Hybrid 14 d | Persistence 14 d |", "|---|---|---|---|---|"]
        for tag, f in cv["folds"].items():
            for s, v in f.items():
                if s.isdigit():
                    md.append(f"| {s} | {v['Hybrid (live setup)']['7']['accuracy']:.3f} | {v['Persistence']['7']['accuracy']:.3f} | "
                              f"{v['Hybrid (live setup)']['14']['accuracy']:.3f} | {v['Persistence']['14']['accuracy']:.3f} |")
        p = cv["pooled"]
        md += ["", f"Pooled over {p['Hybrid (live setup)']['7']['seasons']} held-out seasons:", "",
               "| Model | Lead | Accuracy | Mean over seasons ± SD | Macro-F1 | Alert F1 |", "|---|---|---|---|---|---|"]
        for n, by in p.items():
            for k, m in by.items():
                md.append(f"| {n} | {k} d | {m['accuracy']:.3f} | {m['accuracy_mean_over_seasons']:.3f} ± {m['accuracy_sd_over_seasons']:.3f} | "
                          f"{m['macro_f1']:.3f} | {m['alert_f1']:.3f} |")
        md.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.md").write_text("\n".join(md))
    print(f"Wrote {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
