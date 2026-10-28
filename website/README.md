# CoralWatch website

Landing page and live map for the coral bleaching risk project, focused on the
**northern and central Great Barrier Reef**.

## Open it

Double-click `index.html`. No install or server is needed.

- The saved dates (latest day plus the peak alert day of the 2025, 2024, 2022, 2020, 2017 and 2016 summers) are in `data/snapshots.js`,
  so they load instantly even if NOAA's server is slow.
- Any other date since April 1985 loads live from NOAA when you press **Load** (about 10 seconds).
- Fonts, map tiles and live dates need internet.

## Refresh the data (e.g. the morning of a demo)

```bash
cd website
python3 build_data.py --noaa             # re-download NOAA days (~4 minutes)
python3 build_data.py --noaa 2025-02-20  # optionally add more dates
python3 build_data.py                    # also re-download the reef geography
```

Only the Python standard library is used.

## What the map shows

- NOAA Coral Reef Watch 5 km daily data (v3.1), clipped to the Great Barrier Reef Marine Park
  between 10.4°S and 21°S. Source: PacIOOS ERDDAP mirror, dataset `dhw_5km`.
- Park boundary and 2,722 reef outlines from GBRMPA open data (CC BY 4.0).
- Northern / Central split at 16.5°S. It's an approximate division, not an official boundary.
- Zone cards count the reefs whose nearest 5 km cell is at Alert Level 1 or higher.

| Layer | Variable | Meaning |
|---|---|---|
| Bleaching Alert Area | `CRW_BAA` | 0 No Stress … 4 Alert Level 2 (same scale as `baa_labels.csv`) |
| Degree Heating Weeks | `CRW_DHW` | Heat stress accumulated over 12 weeks (°C-weeks) |
| HotSpot | `CRW_HOTSPOT` | Temperature above the warmest-month average (°C) |
| Temperature anomaly | `CRW_SSTANOMALY` | Difference from the long-term daily average (°C) |
| Sea surface temperature | `CRW_SST` | Daily SST (°C) |

The dataset chart uses the yearly counts from `CNN/baa_labels.csv`.

## Files

- `index.html`, `style.css`: the page
- `app.js`: the live map
- `landing.js`: hero animation, sections, dataset chart
- `build_data.py`: downloads the data into `data/`
