# PhysNet embeddings (daily)

Embeddings from the **new physics-guided CNN-ConvLSTM** (PhysNet, model `max7_s0`), one
256-number vector per day, from **2018-01-01 to 2025-12-31 (2,891 days)**. They replace
`CNN/5_results/Embeddings/cnn_embeddings_daily.csv` and use the same layout, so fusion
code written for the old file works after changing the file name and the column where
`emb_0` starts.

| File | Contents |
|---|---|
| `physnet_embeddings_daily.csv` | date, split, labels and `emb_0` … `emb_255` (one row per day) |
| `physnet_embeddings_daily.npy` | the same `[2891, 256]` float32 matrix, rows in date order |
| `info.json` | model used, dates, cell counts |
| `extract_embeddings.py` | the script that made them |

## Columns

| Column | Meaning |
|---|---|
| `date` | YYYY-MM-DD |
| `split` | `train` / `val` / `test`, the same season split as the model (train 2018–21, val 2022–23, test 2024–25). A season runs Jul–Jun, named by the year it ends |
| `warm_season` | 1 for Nov–Apr (the days scored in our results), else 0 |
| `full_history` | 1 if all 28 input days exist. 0 means some days were missing (e.g. the 2023 gap) and were filled with zeros, so treat those rows with care |
| `baa_label` | the **old** label from `CNN/baa_labels.csv`, kept for comparison only (see the warning below) |
| `alert_level_max` | worst NOAA 7-day-maximum alert level in the Marine Park that day (0–4), from the real values |
| `alert_share` | share of Marine Park cells at Alert Level 1 or higher (0–1) |
| `dhw_mean` | mean Degree Heating Weeks over the Marine Park (°C-weeks) |
| `emb_0` … `emb_63` | mean over all 1,927 Marine Park cells |
| `emb_64` … `emb_127` | max over all park cells (keeps small hot areas visible) |
| `emb_128` … `emb_191` | mean over the Northern GBR (883 cells) |
| `emb_192` … `emb_255` | mean over the Central GBR (1,044 cells) |

## How they're made

For each day, the model reads the last 4 weeks of maps: 7 daily frames and 3 weekly
averages of SST anomaly, HotSpot, DHW, SST, past weather, ENSO/MJO, position and season.
Each reef cell gets 64 numbers: the ConvLSTM memory (32), which summarises the 4 weeks,
plus the encoder output for today (32). These are then pooled into the 4 blocks above.

**No future information.** The vector is taken *before* the layers that see the next
two weeks' weather, so it only contains what was known on that day. You can safely use it
to predict future days.

## Load

```python
import pandas as pd
df = pd.read_csv("forecast/7_embeddings/physnet_embeddings_daily.csv", parse_dates=["date"])
emb_cols = [c for c in df.columns if c.startswith("emb_")]
train = df[(df.split == "train") & (df.full_history == 1)]
X, y = train[emb_cols].values, train.alert_level_max.values
```

## Quality check (linear probe, test seasons 2024–25)

A logistic regression / ridge model on the embeddings alone, trained on the train split
and scored on test, compared with the old CNN embeddings on the same days:

| Task (real NOAA Marine Park values) | Old CNN embeddings | **PhysNet embeddings** |
|---|---|---|
| Share of park at Alert, same day (R²) | 0.62 | **0.89** |
| Worst alert level, same day (accuracy, all days) | 76.5% | **87.7%** |
| Share at Alert, 7 days later (R², all days) | 0.66 | **0.70** |
| Worst alert level, 7 days later (accuracy, all days) | 74.3% | **83.2%** |
| Worst alert level, 7 days later (accuracy, Nov–Apr only) | **76.1%** | 71.5% |

What this means:
- The new embeddings describe the current state of the reef much better.
- For predicting a week ahead, a simple model on either embedding is **not** better than
  "same as today" (87.8% all days, 80.1% Nov–Apr). The full model beats that baseline
  because it also uses the weather forecast and NOAA's DHW formula, which aren't in the
  embedding. So for fusion experiments, compare against the "same as today" baseline,
  not only against the old embeddings.

## Warning about the old `baa_label`

`CNN/baa_labels.csv` **doesn't match** NOAA's alert levels for the Marine Park. On 72 days
it says Alert Level 2 while the whole park is at No Stress, and on 617 days it says Watch
while the park is at No Stress. It probably covers a different area or uses a different
rule. The old CNN embeddings were trained on this label, so they score well on it (about
82%) and worse on the real values. Use `alert_level_max` or `alert_share` as targets.

## Re-make or change

```bash
.venv/bin/python forecast/7_embeddings/extract_embeddings.py                 # model max7_s0 (about 1–2 min)
.venv/bin/python forecast/7_embeddings/extract_embeddings.py --model max7_s1 # another trained run
.venv/bin/python forecast/7_embeddings/extract_embeddings.py --cells         # also per-cell [2891, 1927, 64], about 700 MB
```

Needs `forecast/data/cube.npz` and `forecast/data/weather.npz` (git-ignored; see `forecast/data/README.md`).
