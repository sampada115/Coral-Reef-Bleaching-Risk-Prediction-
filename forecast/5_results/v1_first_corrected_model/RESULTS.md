# Forecast results

**Task:** Bleaching Alert level of every Marine Park cell, 7 days ahead, from the previous 7 days.

**Grid:** 107x106 at 0.1 deg, 1927 park cells.

- **train:** seasons ending 2018-2021, 627 issue days
- **val:** seasons ending 2022-2023, 222 issue days
- **test:** seasons ending 2024-2026, 408 issue days

Model chosen by validation macro_f1: **CNN-ConvLSTM**. Test seasons were not used for training or model choice.

## Test seasons (held out)

| Model | Accuracy | Macro-F1 | Alert precision | Alert recall | Alert F1 | Accuracy on changing cells | Zone % at Alert error (pp) |
|---|---|---|---|---|---|---|---|
| Persistence | 0.656 | 0.562 | 0.554 | 0.576 | 0.565 | 0.000 | 7.84 |
| Climatology | 0.550 | 0.258 | 0.000 | 0.000 | 0.000 | 0.372 | 10.80 |
| Trend + NOAA rule | 0.631 | 0.592 | 0.551 | 0.639 | 0.591 | 0.230 | 7.41 |
| Gradient boosting | 0.642 | 0.563 | 0.568 | 0.587 | 0.577 | 0.433 | 7.35 |
| CNN-ConvLSTM ⭐ | 0.663 | 0.599 | 0.518 | 0.712 | 0.600 | 0.384 | 8.90 |

## Validation seasons

| Model | Accuracy | Macro-F1 | Alert precision | Alert recall | Alert F1 | Accuracy on changing cells | Zone % at Alert error (pp) |
|---|---|---|---|---|---|---|---|
| Persistence | 0.665 | 0.487 | 0.396 | 0.383 | 0.389 | 0.000 | 6.66 |
| Climatology | 0.486 | 0.229 | 0.000 | 0.000 | 0.000 | 0.355 | 6.39 |
| Trend + NOAA rule | 0.577 | 0.500 | 0.433 | 0.538 | 0.480 | 0.204 | 6.28 |
| Gradient boosting | 0.573 | 0.434 | 0.396 | 0.347 | 0.370 | 0.390 | 6.40 |
| CNN-ConvLSTM ⭐ | 0.625 | 0.540 | 0.440 | 0.633 | 0.519 | 0.347 | 6.33 |

Share of cells whose level changes within 7 days (test): 34.4%.

Persistence scores 0 on changing cells by definition; a useful forecast must beat it there without losing much elsewhere.
