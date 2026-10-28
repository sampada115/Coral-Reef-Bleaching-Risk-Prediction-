# 5_results/

| Folder | What it is |
|---|---|
| **`v2_max7_final/`** | **Headline results.** Target: NOAA's 7-day maximum alert. Final model: average of 3 physics-guided models + hybrid |
| `v2_daily_target/` | Same model family on the daily alert level, including all ablations (ocean only, + past weather, + weather forecast, no physics layer, no air temperature) |
| `v1_first_corrected_model/` | First corrected model (CNN-ConvLSTM on real values) vs baselines and gradient boosting |
| `data_checks/` | Checks that the recovered data is correct |

## Files inside `v2_max7_final/` and `v2_daily_target/`

| File | What it shows |
|---|---|
| `RESULTS.md` | All tables: every model at 1, 3, 7 and 14 days; 95% confidence intervals; significance vs "same as today"; reliability |
| `lead_curve.png` | Accuracy, macro-F1 and Alert F1 from 1 to 14 days ahead, with confidence bands |
| `case_2024.png` | Map: observed alert on 3 Mar 2024 vs the forecasts made 3, 7 and 14 days before |
| `reliability.png` | Does "70% chance of Alert" happen about 70% of the time? |
| `ablation.png` | What each part of the model adds |
| `real_forecast_check.json` | Re-test with the weather forecasts actually issued (Jan–Apr 2024 and 2025) |
| `hybrid.json` | Physics persistence vs network for each lead, chosen on validation |
| `results.json` | Every number in machine-readable form |

## Files inside `data_checks/`

| File | What it shows |
|---|---|
| `recovery_check.json` | Values recovered from the images vs NOAA's live server (error ≈ 0.01) |
| `raw_netcdf_check.json` | Values recovered from the images vs your original raw NOAA files (error ≈ 0.01; 97–99% same alert level) |
