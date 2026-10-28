# Forecast results v2 - physics-guided model (daily alert level)

Held-out test seasons 2024-2026 (408 forecast days, 1,927 reef cells each). Brackets: 95% confidence intervals from a moving-block bootstrap (7-day blocks) over forecast days.

**Selected on validation:** PhysNet + weather forecast - ensemble of 3

## 7-day lead

| Model | Accuracy | Macro-F1 | Alert F1 | Alert recall | Within one level |
|---|---|---|---|---|---|
| Persistence | 0.656 [0.604-0.702] | 0.562 [0.521-0.588] | 0.565 [0.462-0.633] | 0.576 [0.474-0.661] | 0.926 [0.899-0.948] |
| Trend + NOAA rule | 0.631 [0.578-0.682] | 0.592 [0.532-0.641] | 0.591 [0.471-0.675] | 0.639 [0.530-0.715] | 0.904 [0.870-0.933] |
| Physics persistence | 0.681 [0.631-0.721] | 0.632 [0.570-0.672] | 0.650 [0.531-0.732] | 0.755 [0.630-0.835] | 0.917 [0.885-0.942] |
| PhysNet (ocean only) (seed 0) | 0.693 [0.644-0.735] | 0.591 [0.520-0.637] | 0.585 [0.446-0.671] | 0.538 [0.369-0.654] | 0.924 [0.894-0.949] |
| PhysNet + observed weather & climate indices (seed 0) | 0.655 [0.590-0.705] | 0.572 [0.509-0.614] | 0.639 [0.505-0.721] | 0.663 [0.489-0.784] | 0.921 [0.891-0.947] |
| PhysNet + weather forecast (seed 0) | 0.774 [0.738-0.808] | 0.705 [0.650-0.740] | 0.673 [0.556-0.757] | 0.549 [0.412-0.656] | 0.949 [0.928-0.967] |
| Same inputs, no DHW formula layer (seed 0) | 0.761 [0.719-0.790] | 0.699 [0.650-0.732] | 0.715 [0.621-0.774] | 0.725 [0.607-0.815] | 0.948 [0.926-0.965] |
| PhysNet + weather forecast without air temperature (seed 0) | 0.746 [0.704-0.781] | 0.700 [0.644-0.732] | 0.715 [0.614-0.778] | 0.641 [0.511-0.733] | 0.949 [0.930-0.964] |
| PhysNet + weather forecast - ensemble of 3 | 0.779 [0.744-0.813] | 0.725 [0.676-0.758] | 0.717 [0.613-0.788] | 0.619 [0.490-0.717] | 0.952 [0.932-0.968] |

## 3-day lead

| Model | Accuracy | Macro-F1 | Alert F1 | Alert recall | Within one level |
|---|---|---|---|---|---|
| Persistence | 0.770 [0.736-0.799] | 0.717 [0.681-0.740] | 0.713 [0.636-0.766] | 0.724 [0.647-0.777] | 0.947 [0.927-0.965] |
| Trend + NOAA rule | 0.764 [0.730-0.798] | 0.735 [0.695-0.768] | 0.727 [0.647-0.788] | 0.754 [0.684-0.809] | 0.940 [0.917-0.961] |
| Physics persistence | 0.783 [0.749-0.813] | 0.754 [0.710-0.786] | 0.757 [0.677-0.817] | 0.812 [0.733-0.871] | 0.945 [0.924-0.962] |
| PhysNet (ocean only) (seed 0) | 0.742 [0.698-0.779] | 0.637 [0.579-0.682] | 0.658 [0.539-0.742] | 0.606 [0.457-0.717] | 0.935 [0.909-0.958] |
| PhysNet + observed weather & climate indices (seed 0) | 0.677 [0.625-0.720] | 0.614 [0.553-0.652] | 0.685 [0.551-0.766] | 0.705 [0.521-0.828] | 0.932 [0.904-0.956] |
| PhysNet + weather forecast (seed 0) | 0.783 [0.750-0.814] | 0.737 [0.685-0.772] | 0.713 [0.605-0.782] | 0.657 [0.520-0.750] | 0.944 [0.922-0.961] |
| Same inputs, no DHW formula layer (seed 0) | 0.760 [0.724-0.795] | 0.684 [0.639-0.715] | 0.676 [0.560-0.755] | 0.746 [0.608-0.848] | 0.933 [0.904-0.957] |
| PhysNet + weather forecast without air temperature (seed 0) | 0.772 [0.736-0.801] | 0.736 [0.687-0.765] | 0.752 [0.655-0.815] | 0.729 [0.592-0.829] | 0.949 [0.928-0.968] |
| PhysNet + weather forecast - ensemble of 3 | 0.797 [0.765-0.825] | 0.754 [0.705-0.787] | 0.737 [0.633-0.801] | 0.716 [0.581-0.805] | 0.946 [0.923-0.963] |

## 1-day lead

| Model | Accuracy | Macro-F1 | Alert F1 | Alert recall | Within one level |
|---|---|---|---|---|---|
| Persistence | 0.871 [0.853-0.888] | 0.843 [0.817-0.860] | 0.840 [0.791-0.874] | 0.843 [0.796-0.876] | 0.968 [0.955-0.978] |
| Trend + NOAA rule | 0.872 [0.852-0.891] | 0.851 [0.823-0.871] | 0.843 [0.792-0.883] | 0.851 [0.804-0.887] | 0.966 [0.952-0.977] |
| Physics persistence | 0.875 [0.857-0.894] | 0.856 [0.830-0.877] | 0.854 [0.805-0.894] | 0.871 [0.822-0.909] | 0.967 [0.955-0.979] |
| PhysNet (ocean only) (seed 0) | 0.793 [0.760-0.824] | 0.701 [0.641-0.742] | 0.718 [0.599-0.790] | 0.645 [0.480-0.756] | 0.946 [0.926-0.964] |
| PhysNet + observed weather & climate indices (seed 0) | 0.695 [0.651-0.733] | 0.638 [0.583-0.675] | 0.705 [0.579-0.787] | 0.708 [0.537-0.830] | 0.935 [0.908-0.957] |
| PhysNet + weather forecast (seed 0) | 0.802 [0.772-0.830] | 0.762 [0.713-0.795] | 0.749 [0.655-0.815] | 0.710 [0.581-0.794] | 0.949 [0.928-0.964] |
| Same inputs, no DHW formula layer (seed 0) | 0.781 [0.746-0.812] | 0.688 [0.640-0.721] | 0.667 [0.544-0.751] | 0.694 [0.518-0.817] | 0.931 [0.903-0.956] |
| PhysNet + weather forecast without air temperature (seed 0) | 0.808 [0.778-0.833] | 0.765 [0.719-0.792] | 0.765 [0.664-0.827] | 0.724 [0.584-0.824] | 0.951 [0.931-0.969] |
| PhysNet + weather forecast - ensemble of 3 | 0.828 [0.803-0.856] | 0.780 [0.739-0.809] | 0.762 [0.662-0.832] | 0.759 [0.623-0.848] | 0.948 [0.927-0.967] |

## 14-day lead

| Model | Accuracy | Macro-F1 | Alert F1 | Alert recall | Within one level |
|---|---|---|---|---|---|
| Persistence | 0.513 [0.448-0.569] | 0.393 [0.352-0.426] | 0.393 [0.299-0.473] | 0.392 [0.276-0.500] | 0.878 [0.839-0.912] |
| Trend + NOAA rule | 0.493 [0.433-0.553] | 0.448 [0.395-0.488] | 0.437 [0.333-0.519] | 0.487 [0.393-0.568] | 0.829 [0.780-0.868] |
| Physics persistence | 0.550 [0.495-0.603] | 0.487 [0.431-0.533] | 0.511 [0.398-0.611] | 0.649 [0.526-0.753] | 0.851 [0.803-0.891] |
| PhysNet (ocean only) (seed 0) | 0.599 [0.542-0.647] | 0.503 [0.445-0.544] | 0.424 [0.314-0.516] | 0.368 [0.253-0.469] | 0.890 [0.849-0.925] |
| PhysNet + observed weather & climate indices (seed 0) | 0.615 [0.550-0.675] | 0.397 [0.345-0.437] | 0.284 [0.183-0.369] | 0.195 [0.116-0.280] | 0.877 [0.834-0.915] |
| PhysNet + weather forecast (seed 0) | 0.739 [0.701-0.774] | 0.670 [0.616-0.707] | 0.639 [0.548-0.711] | 0.522 [0.421-0.609] | 0.939 [0.913-0.960] |
| Same inputs, no DHW formula layer (seed 0) | 0.721 [0.674-0.759] | 0.625 [0.559-0.664] | 0.596 [0.466-0.690] | 0.475 [0.336-0.590] | 0.933 [0.904-0.956] |
| PhysNet + weather forecast without air temperature (seed 0) | 0.665 [0.615-0.706] | 0.583 [0.516-0.621] | 0.570 [0.459-0.642] | 0.439 [0.326-0.521] | 0.924 [0.897-0.951] |
| PhysNet + weather forecast - ensemble of 3 | 0.742 [0.703-0.776] | 0.693 [0.647-0.725] | 0.698 [0.625-0.755] | 0.614 [0.527-0.683] | 0.944 [0.921-0.963] |

## Better than persistence at 7 days? (paired bootstrap, one-sided p-values)

| Model | Accuracy | Macro-F1 | Alert F1 |
|---|---|---|---|
| Trend + NOAA rule | 0.959 | 0.031 | 0.167 |
| Physics persistence | 0.000 | 0.000 | 0.000 |
| PhysNet (ocean only) (seed 0) | 0.008 | 0.133 | 0.273 |
| PhysNet + observed weather & climate indices (seed 0) | 0.528 | 0.381 | 0.029 |
| PhysNet + weather forecast (seed 0) | 0.000 | 0.000 | 0.007 |
| Same inputs, no DHW formula layer (seed 0) | 0.000 | 0.000 | 0.000 |
| PhysNet + weather forecast without air temperature (seed 0) | 0.000 | 0.000 | 0.000 |
| PhysNet + weather forecast - ensemble of 3 | 0.000 | 0.000 | 0.000 |

p < 0.05 means the model is better than persistence with 95% confidence.

## Reliability of the 7-day alert probability

Brier score 0.0372 vs 0.0974 for climatology (skill score 0.618; above 0 = more useful than the long-term average).

| Forecast probability | Mean forecast | Observed frequency | Cells |
|---|---|---|---|
| 0.0-0.1 | 0.01 | 0.01 | 657,796 |
| 0.1-0.2 | 0.15 | 0.21 | 27,148 |
| 0.2-0.3 | 0.25 | 0.33 | 17,891 |
| 0.3-0.4 | 0.35 | 0.47 | 13,226 |
| 0.4-0.5 | 0.45 | 0.57 | 11,272 |
| 0.5-0.6 | 0.55 | 0.70 | 9,156 |
| 0.6-0.7 | 0.65 | 0.78 | 7,558 |
| 0.7-0.8 | 0.75 | 0.83 | 7,638 |
| 0.8-0.9 | 0.85 | 0.86 | 8,848 |
| 0.9-1.0 | 0.97 | 0.93 | 25,683 |

## Validation scores used for model choice (mean of accuracy and macro-F1, 7-day lead)

- PhysNet (ocean only) (seed 0): 0.5890
- PhysNet + observed weather & climate indices (seed 0): 0.5808
- PhysNet + weather forecast (seed 0): 0.7207
- Same inputs, no DHW formula layer (seed 0): 0.7073
- PhysNet + weather forecast (seed 1): 0.7275
- PhysNet + weather forecast (seed 2): 0.7110
- PhysNet + weather forecast without air temperature (seed 0): 0.6595
