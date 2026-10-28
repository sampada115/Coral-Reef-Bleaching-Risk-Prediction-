# Robustness checks of the live model

The live model is unchanged: the PhysNet 7-day-maximum-alert ensemble trained on 2018–2021. Target everywhere: NOAA's 7-day maximum Bleaching Alert level, 1,927 reef cells. Brackets are 95% block-bootstrap confidence intervals.

## 1. Raw NOAA data instead of values decoded from the PNG images

Compared on 2,886 days (2018–2025), every Marine Park cell.

| Variable | PNG colour scale | Decoding error (mean / 99th pct) | Raw values beyond the scale: all days / Nov–Apr |
|---|---|---|---|
| ssta | -3 to 3 | 0.0087 / 0.0339 | 0.0% / 0.0% |
| hotspot | 0 to 6 | 0.0031 / 0.032 | 70.2% / 40.5% |
| dhw | 0 to 12 | 0.0071 / 0.0622 | 0.2% / 0.5% |
| sst | 24 to 32 | 0.0118 / 0.0444 | 8.9% / 0.0% |

Alert level identical on 99.4% of cell-days (worst day 95.0%). Values beyond a scale were clipped in the images; the model's inputs are now clipped to the same ranges (`physnet.INPUT_RANGE`), live and in evaluation, so raw data reach the network exactly as during training.

## 2. Same model, raw data, and the new 2026 season

Decoded data = the inputs used for the published results. Raw = NOAA's values straight from PacIOOS ERDDAP, with ERA5 weather. Season 2026 = Nov 2025 – Apr 2026, never seen in any way.

| Data | Season | Hybrid (live) 7 d | Persistence 7 d | Hybrid 14 d | Persistence 14 d | Brier skill (chance of Alert, 7 d) |
|---|---|---|---|---|---|---|
| Decoded | 2024 | 0.846 [0.809–0.874] | 0.711 [0.633–0.769] | 0.788 [0.734–0.830] | 0.542 [0.434–0.635] | 0.776 |
| Decoded | 2025 | 0.825 [0.771–0.870] | 0.673 [0.590–0.730] | 0.791 [0.736–0.834] | 0.523 [0.412–0.629] | 0.626 |
| Decoded | all | 0.836 [0.803–0.864] | 0.692 [0.636–0.737] | 0.789 [0.750–0.822] | 0.533 [0.463–0.604] | 0.702 |
| Raw | 2024 | 0.855 [0.816–0.884] | 0.732 [0.659–0.802] | 0.794 [0.737–0.838] | 0.571 [0.473–0.670] | 0.778 |
| Raw | 2025 | 0.817 [0.766–0.859] | 0.666 [0.589–0.727] | 0.788 [0.737–0.829] | 0.530 [0.423–0.629] | 0.597 |
| Raw | 2026 | 0.806 [0.775–0.828] | 0.699 [0.651–0.730] | 0.755 [0.712–0.780] | 0.559 [0.496–0.601] | 0.707 |
| Raw | all | 0.826 [0.800–0.846] | 0.699 [0.660–0.731] | 0.779 [0.749–0.803] | 0.553 [0.498–0.602] | 0.697 |
| Raw, unclipped inputs (live system before the fix) | 2024 | 0.851 [0.808–0.884] | 0.732 [0.642–0.799] | 0.789 [0.731–0.833] | 0.571 [0.468–0.671] | 0.779 |
| Raw, unclipped inputs (live system before the fix) | 2025 | 0.810 [0.758–0.853] | 0.666 [0.590–0.728] | 0.782 [0.724–0.827] | 0.530 [0.426–0.623] | 0.600 |
| Raw, unclipped inputs (live system before the fix) | 2026 | 0.816 [0.787–0.837] | 0.699 [0.654–0.731] | 0.763 [0.719–0.792] | 0.559 [0.494–0.598] | 0.707 |
| Raw, unclipped inputs (live system before the fix) | all | 0.826 [0.800–0.846] | 0.699 [0.659–0.734] | 0.778 [0.748–0.801] | 0.553 [0.498–0.601] | 0.698 |

All metrics, raw data, all three test seasons pooled:

| Model | Lead | Accuracy | Macro F1 | Alert F1 |
|---|---|---|---|---|
| Persistence | 1 d | 0.950 [0.942–0.956] | 0.948 [0.940–0.954] | 0.954 [0.941–0.965] |
| Physics persistence | 1 d | 0.974 [0.968–0.980] | 0.976 [0.970–0.981] | 0.985 [0.976–0.991] |
| Ensemble | 1 d | 0.927 [0.915–0.936] | 0.906 [0.890–0.919] | 0.935 [0.912–0.953] |
| Hybrid (live) | 1 d | 0.974 [0.967–0.980] | 0.976 [0.969–0.981] | 0.985 [0.976–0.991] |
| Persistence | 3 d | 0.858 [0.837–0.875] | 0.851 [0.831–0.866] | 0.866 [0.827–0.894] |
| Physics persistence | 3 d | 0.914 [0.894–0.932] | 0.917 [0.897–0.931] | 0.943 [0.917–0.963] |
| Ensemble | 3 d | 0.897 [0.881–0.911] | 0.882 [0.862–0.895] | 0.910 [0.880–0.935] |
| Hybrid (live) | 3 d | 0.914 [0.894–0.931] | 0.917 [0.897–0.931] | 0.943 [0.914–0.963] |
| Persistence | 7 d | 0.699 [0.660–0.731] | 0.679 [0.642–0.706] | 0.702 [0.627–0.763] |
| Physics persistence | 7 d | 0.732 [0.685–0.764] | 0.714 [0.666–0.745] | 0.773 [0.697–0.831] |
| Ensemble | 7 d | 0.826 [0.801–0.846] | 0.811 [0.783–0.833] | 0.813 [0.756–0.863] |
| Hybrid (live) | 7 d | 0.826 [0.800–0.846] | 0.811 [0.782–0.832] | 0.813 [0.751–0.862] |
| Persistence | 14 d | 0.553 [0.498–0.602] | 0.512 [0.468–0.548] | 0.554 [0.479–0.612] |
| Physics persistence | 14 d | 0.594 [0.535–0.642] | 0.548 [0.496–0.592] | 0.624 [0.528–0.707] |
| Ensemble | 14 d | 0.779 [0.750–0.802] | 0.760 [0.729–0.782] | 0.775 [0.713–0.824] |
| Hybrid (live) | 14 d | 0.779 [0.749–0.803] | 0.760 [0.726–0.783] | 0.775 [0.715–0.825] |

## 3. Real weather forecasts at every lead (archived ECMWF runs)

38 issue days (Sundays, 15 Dec – 30 Apr of the 2025 and 2026 seasons). The model is given the ECMWF IFS forecast that was actually issued that day for both forecast weeks, instead of the weather that happened. Bootstrap blocks of 3 weeks.

| Model | Lead | Accuracy | Macro F1 | Alert F1 |
|---|---|---|---|---|
| Persistence | 1 d | 0.937 [0.922–0.949] | 0.940 [0.899–0.951] | 0.952 [0.935–0.969] |
| Physics persistence | 1 d | 0.972 [0.961–0.981] | 0.974 [0.954–0.981] | 0.985 [0.978–0.991] |
| Ensemble + weather that happened (ERA5) | 1 d | 0.913 [0.894–0.931] | 0.900 [0.864–0.925] | 0.937 [0.908–0.964] |
| Ensemble + real ECMWF forecast | 1 d | 0.916 [0.896–0.934] | 0.903 [0.861–0.926] | 0.941 [0.907–0.969] |
| Hybrid (live) + real ECMWF forecast | 1 d | 0.972 [0.961–0.982] | 0.974 [0.953–0.982] | 0.985 [0.979–0.991] |
| Persistence | 3 d | 0.836 [0.802–0.864] | 0.837 [0.742–0.864] | 0.853 [0.786–0.908] |
| Physics persistence | 3 d | 0.919 [0.888–0.948] | 0.923 [0.874–0.943] | 0.956 [0.927–0.976] |
| Ensemble + weather that happened (ERA5) | 3 d | 0.899 [0.880–0.916] | 0.887 [0.835–0.905] | 0.920 [0.884–0.949] |
| Ensemble + real ECMWF forecast | 3 d | 0.900 [0.882–0.917] | 0.887 [0.837–0.907] | 0.925 [0.894–0.949] |
| Hybrid (live) + real ECMWF forecast | 3 d | 0.919 [0.889–0.947] | 0.923 [0.869–0.943] | 0.956 [0.926–0.975] |
| Persistence | 7 d | 0.663 [0.608–0.702] | 0.654 [0.532–0.686] | 0.657 [0.543–0.731] |
| Physics persistence | 7 d | 0.737 [0.685–0.786] | 0.719 [0.625–0.776] | 0.767 [0.649–0.862] |
| Ensemble + weather that happened (ERA5) | 7 d | 0.803 [0.760–0.835] | 0.795 [0.708–0.832] | 0.775 [0.682–0.855] |
| Ensemble + real ECMWF forecast | 7 d | 0.805 [0.764–0.837] | 0.796 [0.702–0.835] | 0.778 [0.686–0.855] |
| Hybrid (live) + real ECMWF forecast | 7 d | 0.805 [0.764–0.836] | 0.796 [0.696–0.833] | 0.778 [0.682–0.852] |
| Persistence | 10 d | 0.590 [0.510–0.635] | 0.571 [0.456–0.608] | 0.552 [0.419–0.656] |
| Physics persistence | 10 d | 0.649 [0.572–0.717] | 0.621 [0.502–0.688] | 0.666 [0.488–0.809] |
| Ensemble + weather that happened (ERA5) | 10 d | 0.791 [0.745–0.828] | 0.778 [0.679–0.817] | 0.755 [0.651–0.841] |
| Ensemble + real ECMWF forecast | 10 d | 0.790 [0.749–0.820] | 0.778 [0.691–0.816] | 0.763 [0.672–0.845] |
| Hybrid (live) + real ECMWF forecast | 10 d | 0.790 [0.750–0.824] | 0.778 [0.696–0.816] | 0.763 [0.674–0.845] |
| Persistence | 14 d | 0.510 [0.415–0.563] | 0.476 [0.340–0.517] | 0.500 [0.380–0.580] |
| Physics persistence | 14 d | 0.569 [0.484–0.633] | 0.521 [0.375–0.590] | 0.565 [0.361–0.724] |
| Ensemble + weather that happened (ERA5) | 14 d | 0.754 [0.705–0.788] | 0.734 [0.618–0.780] | 0.724 [0.612–0.814] |
| Ensemble + real ECMWF forecast | 14 d | 0.727 [0.668–0.777] | 0.711 [0.593–0.761] | 0.710 [0.611–0.798] |
| Hybrid (live) + real ECMWF forecast | 14 d | 0.727 [0.664–0.772] | 0.711 [0.600–0.763] | 0.710 [0.613–0.797] |

One-sided p-values vs persistence (accuracy): 7 d: Ensemble + real ECMWF forecast 0.000, Hybrid (live) + real ECMWF forecast 0.000; 10 d: Ensemble + real ECMWF forecast 0.000, Hybrid (live) + real ECMWF forecast 0.000; 14 d: Ensemble + real ECMWF forecast 0.000, Hybrid (live) + real ECMWF forecast 0.000

Chance of Alert with real forecasts: 7 d Brier skill 0.640; 14 d Brier skill 0.532

## 4. Against NOAA's operational Four-Month Coral Bleaching Outlook

77 weekly Outlook issues (seasons 2024–2026). Target: NOAA 7-day maximum alert for the week ending 14 days after the outlook's initial-condition Sunday (= the Outlook's first forecast week). NOAA's 0.5° Outlook is matched to each reef cell by its nearest ocean pixel.

| Model | Lead | Accuracy | Macro F1 | Alert F1 | Alert Recall |
|---|---|---|---|---|---|
| Persistence | 14 d | 0.547 [0.472–0.607] | 0.506 [0.441–0.547] | 0.551 [0.440–0.633] | 0.562 [0.417–0.681] |
| Physics persistence | 14 d | 0.588 [0.503–0.655] | 0.543 [0.461–0.590] | 0.620 [0.472–0.720] | 0.628 [0.447–0.776] |
| NOAA Outlook (60% map, headline) | 14 d | 0.605 [0.536–0.657] | 0.577 [0.514–0.611] | 0.606 [0.500–0.681] | 0.616 [0.487–0.727] |
| NOAA Outlook (90% map) | 14 d | 0.596 [0.517–0.658] | 0.540 [0.464–0.582] | 0.554 [0.408–0.651] | 0.505 [0.340–0.634] |
| Our ensemble + weather that happened (ERA5) | 14 d | 0.776 [0.734–0.807] | 0.757 [0.700–0.789] | 0.774 [0.675–0.844] | 0.740 [0.617–0.839] |

One-sided p-values that a forecaster is NOT better than NOAA's 60% Outlook: Persistence: acc 0.998, alert F1 0.973; Physics persistence: acc 0.793, alert F1 0.322; NOAA Outlook (90% map): acc 0.794, alert F1 0.992; Our ensemble + weather that happened (ERA5): acc 0.000, alert F1 0.000

Issues that also have an archived ECMWF run (2025–2026, the fully realistic comparison):

| Model | Lead | Accuracy | Macro F1 | Alert F1 | Alert Recall |
|---|---|---|---|---|---|
| Persistence | 14 d | 0.510 [0.418–0.560] | 0.476 [0.339–0.514] | 0.500 [0.371–0.583] | 0.501 [0.357–0.626] |
| Physics persistence | 14 d | 0.569 [0.484–0.633] | 0.521 [0.368–0.579] | 0.565 [0.358–0.719] | 0.565 [0.311–0.773] |
| NOAA Outlook (60% map, headline) | 14 d | 0.572 [0.497–0.616] | 0.566 [0.420–0.606] | 0.567 [0.431–0.673] | 0.577 [0.436–0.698] |
| NOAA Outlook (90% map) | 14 d | 0.570 [0.481–0.640] | 0.524 [0.376–0.580] | 0.498 [0.321–0.626] | 0.445 [0.262–0.596] |
| Our ensemble + weather that happened (ERA5) | 14 d | 0.754 [0.707–0.789] | 0.734 [0.623–0.779] | 0.724 [0.618–0.815] | 0.673 [0.542–0.792] |
| Our ensemble + real ECMWF forecast | 14 d | 0.727 [0.668–0.776] | 0.711 [0.598–0.765] | 0.710 [0.610–0.800] | 0.662 [0.533–0.776] |

Chance of Alert Level 1+ (Brier skill vs climatology): NOAA Outlook 0.181; Our ensemble (ERA5) 0.634 | ECMWF issues: NOAA Outlook 0.048; Our ensemble (ECMWF) 0.532

Check of which week NOAA's 'week 1' describes (its 60% map scored against both readings, accuracy): week_ending_S+14 0.605; week_ending_S+20 0.517

## 5. Against bleaching that was actually seen (2016 and 2017 aerial surveys)

1,396 reef surveys (aims-aerial-2017 71, hughes-aerial-2016 741, hughes-aerial-2017 584). Both summers are before the training data. AUC: 0.5 = no better than chance, 1 = perfect ranking of bleached above unbleached reefs. Bootstrap 95% CIs over surveyed reefs.

| Score | AUC, severe bleaching (>30%) | AUC, moderate or worse (>10%) |
|---|---|---|
| NOAA observed alert on survey day | 0.726 [0.701–0.750] | 0.717 [0.688–0.744] |
| NOAA observed DHW on survey day | 0.790 [0.766–0.815] | 0.769 [0.741–0.797] |
| Our forecast 7 days before: chance of Alert | 0.722 [0.695–0.748] | 0.713 [0.680–0.746] |
| Our forecast 7 days before: level | 0.706 [0.680–0.732] | 0.693 [0.667–0.723] |
| Our forecast 14 days before: chance of Alert | 0.677 [0.649–0.707] | 0.661 [0.628–0.695] |
| Our forecast 14 days before: level | 0.632 [0.605–0.659] | 0.607 [0.575–0.635] |
| Same as today, 7 days before | 0.720 [0.694–0.744] | 0.695 [0.668–0.721] |
| Same as today, 14 days before | 0.756 [0.732–0.781] | 0.723 [0.694–0.748] |
| Physics persistence, 14 days before | 0.707 [0.680–0.731] | 0.674 [0.646–0.702] |

## 6. Calibrated chance of Alert

temperature scaling per lead (minimum log-loss), fitted on validation seasons 2022-2023 only; an extra isotonic correction was tested and rejected (worse on test at leads 9-14).

| Lead | T | Brier before | after | ECE before | after | Isotonic (rejected) Brier |
|---|---|---|---|---|---|---|
| 1 d | 0.882 | 0.0179 | 0.0177 | 0.0054 | 0.003 | 0.0183 |
| 2 d | 0.936 | 0.0179 | 0.0179 | 0.0041 | 0.0029 | 0.018 |
| 3 d | 0.994 | 0.0215 | 0.0215 | 0.0052 | 0.0052 | 0.0213 |
| 4 d | 1.024 | 0.0262 | 0.0262 | 0.0067 | 0.0067 | 0.026 |
| 5 d | 1.024 | 0.0304 | 0.0303 | 0.007 | 0.005 | 0.0302 |
| 6 d | 1.055 | 0.0357 | 0.0356 | 0.0105 | 0.0083 | 0.0356 |
| 7 d | 1.055 | 0.0409 | 0.0406 | 0.0145 | 0.0124 | 0.0405 |
| 8 d | 1.087 | 0.0444 | 0.044 | 0.0161 | 0.0121 | 0.044 |
| 9 d | 1.154 | 0.0469 | 0.0461 | 0.0187 | 0.0124 | 0.0469 |
| 10 d | 1.189 | 0.0482 | 0.0473 | 0.018 | 0.0101 | 0.0491 |
| 11 d | 1.189 | 0.0492 | 0.0484 | 0.0182 | 0.0105 | 0.0505 |
| 12 d | 1.225 | 0.0501 | 0.0494 | 0.0162 | 0.008 | 0.0532 |
| 13 d | 1.225 | 0.0501 | 0.0497 | 0.0134 | 0.0063 | 0.0542 |
| 14 d | 1.189 | 0.0514 | 0.0511 | 0.0131 | 0.0073 | 0.0565 |

Figure: `../v2_max7_final/reliability_calibrated.png`.
