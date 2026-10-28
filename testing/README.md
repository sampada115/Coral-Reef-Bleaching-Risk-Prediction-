# testing/

All the testing for CoralWatch: test code, results, charts and the slide content.

| File | What it is |
|---|---|
| [SLIDES.md](SLIDES.md) | **Slide content** for Implementation Details, Test Plan & Strategy, Results & Discussion |
| [TEST_REPORT.md](TEST_REPORT.md) | Every test case with its result, the performance numbers, the LLM tests and the defect log |
| `run_all.sh` | Runs everything and writes `reports/` and `charts/` |
| `conftest.py` | Shared test setup |
| `test_integration.py` | Integration tests: data → model → API → AI briefing → website (7) |
| `test_model_validation.py` | Research checks: data integrity, splits, **no leakage**, reproducibility, results vs baselines (11) |
| `test_robustness.py` | Fault injection: NOAA, weather and LLM down; hallucinated output (6) |
| `test_security.py` | Input validation, path traversal, secrets, prompt injection, XSS (24) |
| `perf_test.py` | Response time, concurrent load and memory on the real server |
| `llm_adversarial.py` | Tricky questions for the free local LLM (prompt injection, planted facts, off-topic) |
| `make_charts.py` | Builds the slide charts |
| `charts/` | 10 slide images: architecture, result charts, 2024 case study, website screenshots |
| `reports/` | Raw outputs: pytest, JUnit XML, coverage HTML, Bandit, pip-audit, performance, LLM tests |

The 16 unit tests stay in `tests/test_core.py`; `run_all.sh` runs both folders.

## Run

```bash
./testing/run_all.sh          # all automated tests, security scans, performance test, charts (~3 min)
./testing/run_all.sh --llm    # + LLM tests (needs `ollama serve`, ~15 min)
```

Single parts:

```bash
.venv/bin/python -m pytest tests testing -q          # 64 tests
.venv/bin/python testing/perf_test.py
.venv/bin/python testing/llm_adversarial.py
```

Needs `forecast/data/` (git-ignored) for the model-validation tests; they're skipped without it.
Test tools: `.venv/bin/pip install pytest-cov bandit pip-audit psutil`.
