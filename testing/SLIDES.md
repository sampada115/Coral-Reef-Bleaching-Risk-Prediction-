# Presentation content: 3 sections

Copy each slide's text into the deck and add the image named under **🖼 Add image**.
All images are in **`testing/charts/`**. Speaker notes are what to say; they don't go on the slide.

Fill in yourselves: **[Name 1–4]** (team members and roles) on slide 2.4.

## All images at a glance

| Image file | Goes on slide |
|---|---|
| `0_architecture.png` | 1.1 System architecture |
| `7_website_landing.png` | 1.1 (or a title / demo slide) |
| `8_website_forecast_map.png` | 1.3 Model output on the live site |
| `9_website_ai_briefing.png` | 1.4 Fact-checked AI briefing |
| `5_performance_before_after.png` | 2.5 Benefits / testing outcome |
| `1_accuracy_vs_lead.png` | 3.1 Results |
| `6_case_study_2024.png` | 3.2 2024 case study |
| `2_method_comparison_7d.png` | 3.3 Comparison with existing methods |
| `4_real_forecast_test.png` | 3.4 Realistic test |
| `3_ablation.png` | 3.4 Ablation |

---
---

# 1. IMPLEMENTATION DETAILS

## Slide 1.1 · System architecture and modules

🖼 **Add image:** `0_architecture.png` (main), and optionally `7_website_landing.png` small in a corner.

| Module | Technology used | Role |
|---|---|---|
| 1. Data recovery | Python, NumPy, SciPy KD-tree, Pillow | Coloured NOAA images → real values |
| 2. Weather & climate | Open-Meteo API, NOAA CPC, BoM | Wind, cloud, sunlight, air temperature; ENSO, MJO |
| 3. Dataset builder | NumPy | Season split, NOAA 7-day-maximum labels |
| 4. PhysNet forecaster | PyTorch: CNN + ConvLSTM, Apple MPS | Alert level for 1,927 reef cells, 1–14 days ahead |
| 5. Evaluation + hybrid | scikit-learn, NumPy | Baselines, bootstrap CIs, physics/network choice per lead |
| 6. Backend API | FastAPI, Uvicorn | Live NOAA data every 3 h; forecast and AI endpoints |
| 7. Report search (NLP) | BM25, pypdf | 3,180 report passages, no hindsight |
| 8. AI briefing | Ollama + Qwen 2.5 3B (free, local) | Cited explanation, fact-checked |
| 9. Website | HTML/CSS/JS, Leaflet | Landing page, live map, forecast layer, AI panel |

> **Speaker notes:** Four layers: data → processing → model → live system. All tools are
> open-source and all data is free. The model has about 210,000 parameters and trains on an 8 GB laptop.

---

## Slide 1.2 · Algorithm 1: recovering real values from images

**Problem:** the original pipeline turned colour maps grey, so +3 °C and −3 °C looked the same.
**Idea:** each pixel was painted from a fixed colour scale, so the colour gives the exact value back.

```
Algorithm RecoverValues(image, colormap, vmin, vmax)
  table ← the 256 colours of colormap          // viridis or RdBu_r
  tree  ← KD-tree(table)
  for each pixel p:
      i ← nearest colour index to p in tree     // nearest-neighbour search
      value[p] ← vmin + (i + 0.5) / 256 × (vmax − vmin)
  return value
```

**Interpretation:**
- 2,891 days recovered.
- Mean error 0.006–0.018 compared with NOAA's raw files.
- The alert level matches NOAA on 97–99% of cells.

> **Speaker notes:** This avoids re-downloading 19 GB from NOAA, whose servers block bulk
> downloads. One colour step is about 0.02 °C, far smaller than what changes an alert level.

---

## Slide 1.3 · Algorithm 2: PhysNet, the physics-guided CNN-ConvLSTM

🖼 **Add image:** `8_website_forecast_map.png` (the model's 7-day forecast for the 2024 event on the live site)

```
Input: 10 maps (last 7 days + 3 weekly means) × 16 channels
1. f_k ← CNN(map_k)                         spatial features of each map
2. h   ← ConvLSTM(f_1 … f_10)               memory of the last 4 weeks
3. z   ← Conv(h, f_10, DHW today, drop-off, 14-day weather forecast)
4. HS_k  ← HotSpot head(z)                  k = 1 … 14 days
5. DHW_k ← DHW_today + Σ HS⁺/7 − drop-off_k     ← NOAA formula layer
6. P(level_k) ← softmax(ClassHead(z, HS_k, DHW_k))
Loss = cross-entropy + 0.1·ordinal + 0.5·Huber(HotSpot)
Final = average of 3 models; days 1–4 use physics (chosen on validation)
```

**Interpretation:** the network learns only the uncertain part, future HotSpot. NOAA's own
formula then turns it into DHW and the alert level, so the physics is never "unlearned".

---

## Slide 1.4 · Algorithms 3–4: report search and fact-checked AI briefing

🖼 **Add image:** `9_website_ai_briefing.png` (cited briefing with the "✓ Fact-checked 14/14 numbers" badge)

```
Algorithm GroundedBriefing(date, zone, question)
  facts   ← NOAA numbers + forecast per lead (with date) + model accuracy
  sources ← BM25(reports published BEFORE date)     // no hindsight
            keep ≥ 35% of best score, ≤ 2 per report
  answer  ← LLM(facts, sources, "use only these; tag every sentence")
  for each sentence s in answer:
      number not in facts/sources        → delete s
      date + number not stated together  → delete s
      factual claim without a source tag → tag if supported, else delete
  return answer + fact-check badge
```

**Interpretation:** the AI can only repeat checked facts; anything unverifiable is removed
before the user sees it. It's free and runs locally.

---
---

# 2. TEST PLAN AND STRATEGY

## Slide 2.1 · Testing activities and timeline

| When (2026) | Testing activity |
|---|---|
| Aug | Review of the original pipeline: found test-set leakage, no baseline, grey-image problem |
| Sep, weeks 3–4 | **Data validation:** recovered values vs NOAA raw files; NOAA alert and DHW rules re-checked |
| Sep, week 4 | **Model evaluation** on held-out seasons: baselines, confidence intervals, significance tests, ablations |
| 30 Sep – 1 Oct | **Unit tests** (16) written with the modules; AI grounding evaluation |
| 12 Oct | **Leakage test** of the embeddings for the fusion experiments |
| 23 Oct | **Integration, robustness, security, performance and LLM adversarial testing**; 4 defects fixed and re-tested |

> **Speaker notes:** Testing ran alongside development. For the research part we followed
> standard practice: train on 2018–21, choose the model on 2022–23, and test once on 2024–25.

---

## Slide 2.2 · Test methods followed, and why

| | Method | Why we used it | Tests |
|---|---|---|---|
| **Functional** | Unit testing | NOAA's alert rule, DHW formula and filters must be exact | 16 |
| | Integration testing | Data → model → API → AI briefing must agree end to end | 7 |
| | Model validation *(research)* | No data leakage, disjoint splits, reproducible, beats baselines | 11 |
| | Robustness (fault injection) | NOAA, the weather service or the LLM can go offline | 6 |
| **Non-functional** | Security testing | Bad input, path traversal, leaked secrets, prompt injection, XSS | 24 |
| | Performance / load testing | Response time, 10–50 concurrent users, memory | 3 load levels |
| | LLM hallucination testing | Adversarial questions; every number must be verified | 7 + 8 briefings |
| | Statistical evaluation *(research)* | Held-out seasons, 95% bootstrap CI, paired p-values | 408 days |

> **Speaker notes:** Since this is a research project, "does the model really work" is tested
> as seriously as "does the code work". Example: we scramble the next 14 days of ocean data
> and check that today's forecast doesn't change. That proves the model can't see the answer.

---

## Slide 2.3 · Test tools: automated and open-source

| Tool | Used for | Type |
|---|---|---|
| **pytest** + FastAPI TestClient | Unit, integration, validation, robustness, security tests | Automated, open-source |
| **pytest-cov** | Code coverage (65% overall; 72–100% on core modules) | Automated, open-source |
| **Bandit** | Static security analysis of the Python code | Automated, open-source |
| **pip-audit** | Known vulnerabilities in the packages we use | Automated, open-source |
| **httpx + psutil** | Load test and memory measurement | Automated, open-source |
| **NumPy / scikit-learn** | Bootstrap confidence intervals, significance tests | Open-source |
| **Ollama** | Runs the LLM locally for the hallucination tests | Open-source |
| **Matplotlib** | Result charts | Open-source |

**One command runs everything:** `./testing/run_all.sh` (64 tests in under a minute, plus the scans and load test).

---

## Slide 2.4 · Test environment and team roles

**Environment**
- MacBook: 8 cores, 8 GB RAM, Apple MPS GPU, macOS.
- Python 3.14, PyTorch 2.14, FastAPI 0.141.
- Local server on 127.0.0.1 and Ollama 0.20 with Qwen 2.5 3B.
- Cached NOAA data, so tests also run offline.

| Team member | Role in testing |
|---|---|
| **[Name 1]** | [e.g. data validation, model evaluation and statistics] |
| **[Name 2]** | [e.g. backend: integration, robustness and performance tests] |
| **[Name 3]** | [e.g. AI briefing: hallucination tests; security testing] |
| **[Name 4]** | [e.g. website: manual browser testing; test report and documentation] |

---

## Slide 2.5 · Benefits and drawbacks of this approach

🖼 **Add image:** `5_performance_before_after.png`

**Benefits**
- Automated and repeatable: 64 tests run in under a minute after every change.
- **Found and fixed 4 real defects:**
  - one missing NOAA day broke every forecast when NOAA was offline;
  - an invalid date crashed the server;
  - map requests were slow;
  - model files were loaded with an unsafe setting.
- Testing made the system faster: map data 178 → 1.2 ms; first forecast 84 → 7.5 s.
- Research rigour: leakage and significance are tested, not assumed.

**Drawbacks**
- Only 2 test seasons and one region.
- We predict NOAA heat-stress alerts; checking against observed bleaching surveys is still pending.
- The website UI is tested manually (no browser automation).
- The small 3B LLM: facts are checked automatically, writing quality is judged by people.

---
---

# 3. RESULTS AND DISCUSSION

## Slide 3.1 · Results after testing: forecast accuracy

🖼 **Add image:** `1_accuracy_vs_lead.png`

| Days ahead | PhysNet (ours) | "Same as today" baseline | Gain |
|---|---|---|---|
| 1 | 97.1% | 94.7% | +2.4 |
| 3 | 90.6% | 85.0% | +5.6 |
| **7** | **82.4%** [79.6–84.9] | 68.7% | **+13.7** |
| 14 | 76.7% | 50.8% | +25.9 |

7 days ahead:
- Alert F1 0.83.
- 95% of forecasts are right or off by just one level.
- Brier skill 0.73.
- **p < 0.001** against the baseline.

> **Speaker notes:** Test seasons 2024–25 were never used for training or model choice. The
> further ahead we forecast, the bigger our advantage, because "same as today" goes out of date quickly.

---

## Slide 3.2 · Case study: the 2024 mass bleaching

🖼 **Add image:** `6_case_study_2024.png`

- The forecast issued **7 days before** 3 Mar 2024 shows the same pattern as what NOAA observed:
  - Alert Level 2 in the north;
  - a Watch area in the central inshore.
- Even **14 days** ahead the overall pattern holds; the details blur.
- Live spot check for the week to 10 Mar 2024 (reefs at Alert Level 1 or higher):

  | | Northern GBR | Central GBR |
  |---|---|---|
  | Forecast | 748 | 1,010 |
  | Observed | 744 | 873 |

---

## Slide 3.3 · Comparison with existing methods

🖼 **Add image:** `2_method_comparison_7d.png`

| Method (7 days ahead, same test days) | Accuracy |
|---|---|
| "Same as today" (persistence, the standard baseline) | 68.7% |
| Physics persistence (NOAA's DHW formula, no learning) | 71.9% |
| Trend + NOAA rule | 72.0% |
| Plain CNN-ConvLSTM, no physics or weather (our v1)* | 66.3% vs 65.6% baseline |
| Gradient boosting (our v1)* | 64.2% vs 65.6% baseline |
| **PhysNet ensemble (ours)** | **82.4%** |

\*daily-target version, compared with its own baseline

> **Speaker notes:**
> - Plain deep learning barely beat the baseline. Physics plus weather forecasts made the difference.
> - Published coral work is mostly **detection** from photos or satellite images. We found no published
>   day-ahead, reef-level accuracy to compare against directly, so we compare with standard
>   forecasting baselines on exactly the same days.

---

## Slide 3.4 · Realistic test and what each part contributes

🖼 **Add images:** `4_real_forecast_test.png` (left) and `3_ablation.png` (right)

- **Realistic test:** using the weather forecasts actually issued at the time (Jan–Apr 2024 and 2025):
  **76.9%** vs 63.8%. The model stays useful with imperfect forecasts.
- **Ablation (7 days, daily target):**
  - ocean data only: 69.3%;
  - adding the weather forecast: **77.4%** (the biggest gain);
  - ensemble of 3: 77.9%.
- Removing the **DHW formula layer** drops accuracy to 76.1%, and the drop is largest at 14 days.
- Past weather alone and training with forecast error did **not** help. We report both honestly.

---

## Slide 3.5 · Are the results as expected? Deviations and reasons

| Initial plan / estimate | Actual result | Reason for the change |
|---|---|---|
| ≥ 85% at 7 days (original reported 83.75%) | **82.4%** (97.1% at 1 day) | The original 83.75% was inflated: the test set was used to pick the model and there was no baseline (a trivial rule got 81.1%). Our task is also harder: 1,927 reef cells instead of one regional label |
| Daily alert level as target | NOAA 7-day maximum alert | Daily levels are noisy (even a 1-day "same as today" gets only 87%). The 7-day maximum is NOAA's official product |
| CNN-LSTM on grey images | PhysNet on real values | Grey images lost the sign of the anomaly; we recovered the values from the colours |
| Download raw NOAA data | Recovered from our own images | NOAA's servers throttled and blocked bulk downloads |
| NLP fused into prediction (reported 90.1%) | NLP grounds the AI explanations | Report embeddings had only 7 distinct vectors (one per year) and risked hindsight |
| Paid LLM (Claude) | Free local model (Qwen 2.5) | Cost; the fact-checker keeps it accurate |
| Forecast-error training would help | It didn't | Reported as a negative result |

> **Speaker notes:** The headline number went down slightly but became honest. Before, the
> model beat a simple rule by about 2.6 points; now it beats a proper baseline by 13.7.

---

## Slide 3.6 · Results of system testing

| Area | Result |
|---|---|
| Automated tests | **64/64 passed** (unit 16, integration 7, model validation 11, robustness 6, security 24) |
| No-leakage test | Forecast and embeddings unchanged when future data is scrambled ✔ |
| Security | Bandit: 0 high-severity issues (1 medium fixed); pip-audit: 0 known vulnerabilities |
| Performance | 50 concurrent users at **408 requests/s**, 0 errors; map data **1.2 ms** |
| AI hallucination | Adversarial questions **7/7 passed** (prompt injection blocked); 8 briefings over 4 years: **47/47 numbers verified, 37/37 sentences cited** |

---

## Slide 3.7 · Discussion and conclusion

- A **physics-guided** model beats plain deep learning and every standard baseline, and the gain grows with lead time.
- **Weather forecasts** are the biggest single driver of skill a week ahead.
- An honest evaluation: **82.4%** at 7 days, 13.7 points above the baseline and statistically significant.
- The AI explanations are **grounded**: every number shown is checked against data or reports.
- **Future work:**
  - validate against observed bleaching (2020 aerial surveys);
  - add more seasons and regions;
  - compare with NOAA's operational outlooks.
