#!/usr/bin/env bash
# Robustness checks, end to end (see forecast/5_results/v2_robustness/RESULTS.md):
#   waits for the raw NOAA / ECMWF / Outlook downloads, builds cube_raw.npz, then runs the
#   evaluations and the observed-bleaching check; with --cv also the leave-seasons-out
#   cross-validation (~12-15 h on an 8 GB MacBook).
#     ./forecast/run_robustness.sh [--cv]      (logs in forecast/logs/robustness_*.log)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=forecast/logs

while pgrep -f "download_raw_cube|download_ecmwf_runs|download_noaa_outlook" >/dev/null; do sleep 60; done
$PY forecast/1_data_preparation/download_raw_cube.py --start 2015-09-01 > $LOG/robustness_build.log 2>&1   # fetches anything missed, then builds
$PY forecast/4_evaluation/evaluate_extended.py > $LOG/robustness_eval.log 2>&1
$PY forecast/4_evaluation/validate_bleaching.py > $LOG/robustness_bleaching.log 2>&1
$PY forecast/4_evaluation/report_robustness.py
if [ "${1:-}" = "--cv" ]; then
  $PY forecast/3_training/cross_validate.py > $LOG/robustness_cv.log 2>&1
  $PY forecast/4_evaluation/report_robustness.py
fi
echo "robustness checks finished"
