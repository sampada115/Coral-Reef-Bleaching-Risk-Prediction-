import os
import glob
import xarray as xr
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image

# Years and months to process
YEARS = [2018, 2019]
MONTHS = range(1, 13)

# All variables to process
VARIABLES = ["dhw", "hotspot", "sst", "ssta"]

VAR_CONFIG = {

    "dhw": {
        "nc_var": "degree_heating_week",
        "cmap": cm.viridis,
        "vmin": 0,
        "vmax": 12,
    },

    "hotspot": {
        "nc_var": "hotspot",
        "cmap": cm.viridis,
        "vmin": 0,
        "vmax": 6,
    },

    "sst": {
        "nc_var": "analysed_sst",
        "cmap": cm.viridis,
        "vmin": 24,
        "vmax": 32,
    },

    "ssta": {
        "nc_var": "CRW_SSTANOMALY",
        "cmap": cm.RdBu_r,
        "vmin": -3,
        "vmax": 3,
    },
}

skipped_files = []  # track all skipped files

for VARIABLE in VARIABLES:

    print("\n==============================")
    print("Processing variable:", VARIABLE)
    print("==============================")

    cfg    = VAR_CONFIG[VARIABLE]
    cmap   = cfg["cmap"]
    norm   = mcolors.Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])
    nc_var = cfg["nc_var"]

    for YEAR in YEARS:
        for MONTH in MONTHS:

            IN_DIR  = f"1_raw_netcdf/{VARIABLE}/{YEAR}/{MONTH:02d}"
            OUT_DIR = f"2_images/{VARIABLE}/{YEAR}/{MONTH:02d}"

            if not os.path.exists(IN_DIR):
                print("Skipping missing folder:", IN_DIR)
                continue

            os.makedirs(OUT_DIR, exist_ok=True)

            files = sorted(glob.glob(f"{IN_DIR}/*.nc"))
            print(f"\nProcessing {VARIABLE} {YEAR}-{MONTH:02d} | Files: {len(files)}")

            for f in files:
                try:

                    # ── Open file ──────────────────────────────────
                    try:
                        ds = xr.open_dataset(f, engine="netcdf4")
                    except Exception:
                        ds = xr.open_dataset(f, engine="scipy")

                    # ── Get variable ───────────────────────────────
                    try:
                        data = ds[nc_var]
                    except KeyError:
                        available = list(ds.data_vars)
                        print(f"  ⚠️  '{nc_var}' not found → using: {available[0]}")
                        data = ds[available[0]]

                    # ── Handle time dimension ──────────────────────
                    if "time" in data.dims:
                        data = data.isel(time=0)

                    # ── Detect lat/lon coordinate names ───────────
                    lat_name = "lat"      if "lat"      in data.coords else "latitude"
                    lon_name = "lon"      if "lon"      in data.coords else "longitude"

                    # ── Read lat values before crop ────────────────
                    lat_vals = data[lat_name].values

                    # ── Crop to GBR (handle ascending/descending lat)
                    if lat_vals[0] < lat_vals[-1]:
                        # ascending → slice low to high
                        data = data.sel(
                            {lat_name: slice(-24, -10),
                             lon_name: slice(142, 154)}
                        )
                    else:
                        # descending → slice high to low
                        data = data.sel(
                            {lat_name: slice(-10, -24),
                             lon_name: slice(142, 154)}
                        )

                    # ── Check crop result is not empty ─────────────
                    if data.size == 0:
                        print(f"  ⚠️  Empty after crop: {os.path.basename(f)}")
                        skipped_files.append({"variable": VARIABLE, "file": os.path.basename(f)})
                        continue

                    # ── Kelvin → Celsius for SST ───────────────────
                    if VARIABLE == "sst":
                        if float(np.nanmax(data.values)) > 100:
                            data = data - 273.15

                    arr = data.values

                    # ── Check valid data exists ────────────────────
                    if np.all(np.isnan(arr)):
                        print(f"  ⚠️  All NaN: {os.path.basename(f)}")
                        skipped_files.append({"variable": VARIABLE, "file": os.path.basename(f)})
                        continue

                    # ── Land mask (NaN = land/missing) ─────────────
                    land_mask = np.isnan(arr)

                    arr = np.nan_to_num(arr, nan=cfg["vmin"])
                    arr = np.clip(arr, cfg["vmin"], cfg["vmax"])

                    # ── Apply colormap ─────────────────────────────
                    rgba = cmap(norm(arr))
                    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)

                    # ── Paint land black ───────────────────────────
                    rgb[land_mask] = [0, 0, 0]

                    # ── Flip if latitude is ascending (north = top) ─
                    if lat_vals[0] < lat_vals[-1]:
                        rgb = np.flipud(rgb)

                    # ── Save PNG ───────────────────────────────────
                    base = os.path.basename(f).replace(".nc", "")
                    out  = f"{OUT_DIR}/{base}.png"

                    Image.fromarray(rgb).save(out)
                    print(f"  ✓ Saved: {os.path.basename(out)} | shape: {rgb.shape}")

                    ds.close()

                except Exception as e:
                    print(f"  ⚠️  Skipping {os.path.basename(f)}: {str(e)[:80]}")
                    skipped_files.append({
                        "variable": VARIABLE,
                        "file": os.path.basename(f)
                    })
                    continue

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "="*50)
print("✓ All done!")
if skipped_files:
    print(f"\n⚠️  Skipped {len(skipped_files)} corrupted/missing files:")
    for s in skipped_files:
        print(f"   [{s['variable']}] {s['file']}")
else:
    print("No files were skipped — perfect!")