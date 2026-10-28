#!/usr/bin/env bash
# Runs every CoralWatch test and writes the reports to testing/reports/.
#   ./testing/run_all.sh            all tests except the LLM test
#   ./testing/run_all.sh --llm      also the LLM hallucination test (needs: ollama serve)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=testing/reports
mkdir -p "$R"

echo "== 1. Unit + integration + validation + robustness + security tests (pytest, with coverage)"
$PY -m pytest tests testing -q -p no:warnings \
  --junitxml="$R/junit.xml" \
  --cov=backend --cov=coralwatch --cov=forecast/2_model --cov-report=term --cov-report=html:"$R/coverage_html" \
  | tee "$R/pytest.txt"

echo "== 2. Static security analysis (Bandit)"
.venv/bin/bandit -r backend coralwatch forecast/1_data_preparation forecast/2_model forecast/3_training \
  forecast/4_evaluation forecast/7_embeddings -f txt -o "$R/bandit.txt" -q
grep -A4 "Total issues (by severity)" "$R/bandit.txt"

echo "== 3. Dependency vulnerability audit (pip-audit)"
.venv/bin/pip-audit -r requirements.txt --progress-spinner off | tee "$R/pip_audit.txt"

echo "== 4. Performance and load test"
$PY testing/perf_test.py

if [ "${1:-}" = "--llm" ]; then
  echo "== 5. LLM hallucination / adversarial test (local model)"
  $PY testing/llm_adversarial.py
  echo "== 6. LLM grounding across dates"
  $PY -m backend.ai.eval_grounding --dates 2020-03-08 2022-03-12 2024-03-03 2025-03-09
  cp backend/data/grounding_eval.json "$R/grounding_eval.json"
fi

echo "== Charts"
$PY testing/make_charts.py
echo "Done. Reports in $R/, charts in testing/charts/"
