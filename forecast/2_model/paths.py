"""Every file location used by the forecast pipeline, in one place.

Scripts in the numbered folders add forecast/2_model and the project root to sys.path,
then `from paths import ...`.
"""

from pathlib import Path

FORECAST = Path(__file__).resolve().parents[1]          # forecast/
ROOT = FORECAST.parent                                   # project root

# data
DATA = FORECAST / "data"
CUBE = DATA / "cube.npz"                                 # 2,891 days x [SST anomaly, HotSpot, DHW, SST]
CUBE_V1 = DATA / "cube_season_v1.npz"                    # earlier warm-season-only dataset (v1)
WEATHER = DATA / "weather.npz"                           # weather at 74 points + ENSO/MJO
WEATHER_EXT = DATA / "weather_ext.npz"                   # same, extended into 2026 (raw-data evaluation, cross-validation)
FORECAST_ERROR_STATS = DATA / "forecast_error_stats.json"
RAW_NOAA = DATA / "raw_noaa_partial"                     # partial raw download (NOAA throttles)
RAW_DAYS = DATA / "raw_days"                             # raw NOAA values per month (PacIOOS, 0.1 deg), 2015-09 onwards
CUBE_RAW = DATA / "cube_raw.npz"                         # same layout as CUBE, built from RAW_DAYS (no image decoding)
WEATHER_CACHE = DATA / "weather_cache"
PREVRUNS_CACHE = DATA / "prevruns_cache"                 # archived weather forecasts, leads 1-7 (Previous Runs API)
ECMWF_RUNS = DATA / "ecmwf_runs"                         # archived full ECMWF IFS runs, leads 1-15 (Single Runs API)
NOAA_OUTLOOK = DATA / "noaa_outlook"                     # NOAA CRW Four-Month Coral Bleaching Outlook (v5) NetCDFs
BLEACHING_DB = DATA / "bleaching"
IMAGES = ROOT / "CNN" / "2_images"                       # the original PNG maps
SNAPSHOTS_JS = ROOT / "website" / "data" / "snapshots.js"

# results
RESULTS = FORECAST / "5_results"
RESULTS_FINAL = RESULTS / "v2_max7_final"                # headline: NOAA 7-day maximum alert
RESULTS_DAILY = RESULTS / "v2_daily_target"              # daily alert level + ablations
RESULTS_V1 = RESULTS / "v1_first_corrected_model"
DATA_CHECKS = RESULTS / "data_checks"

# models
MODELS = FORECAST / "6_saved_models"
MODELS_V1 = MODELS / "v1"
MODELS_V2 = MODELS / "v2"

LOGS = FORECAST / "logs"
