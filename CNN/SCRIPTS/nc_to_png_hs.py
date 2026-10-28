import os
import glob
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
HS_VAR = "hotspot"
HS_MIN = 0.0
HS_MAX = 6.0

for YEAR in [2018, 2019]:
    for MONTH in range(1, 13):

        IN_DIR = f"1_raw_netcdf/hotspot/{YEAR}/{MONTH:02d}"
        OUT_DIR = f"2_images/hotspot/{YEAR}/{MONTH:02d}"
        os.makedirs(OUT_DIR, exist_ok=True)

        files = sorted(glob.glob(os.path.join(IN_DIR, "*.nc")))
        if not files:
            print(f"⚠️ No HotSpot files for {YEAR}-{MONTH:02d}")
            continue

        print(f"\n🖼️ Converting HotSpot {YEAR}-{MONTH:02d} | Files: {len(files)}")

        for f in files:
            try:
                ds = xr.open_dataset(f)
                hotspot = ds[HS_VAR]

                if "time" in hotspot.dims:
                    hotspot = hotspot.isel(time=0)

                # Crop GBR
                hotspot = hotspot.sel(
                    lat=slice(-10, -24),
                    lon=slice(142, 154)
                )

                # Mask invalid values
                hotspot = hotspot.where(hotspot >= 0)

                # Extract date from filename: hotspot_YYYY_MM_DD.nc
                base = os.path.basename(f).replace(".nc", "")
                _, y, m, d = base.split("_")

                out_path = f"{OUT_DIR}/hotspot_{y}_{m}_{d}.png"

                plt.figure(figsize=(8, 6))
                hotspot.plot(
                    cmap="viridis",
                    vmin=HS_MIN,
                    vmax=HS_MAX,
                    cbar_kwargs={"label": "HotSpot (°C)"}
                )
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_path, dpi=250)
                plt.close()

                print("✅ Saved:", out_path)

            except Exception as e:
                print("❌ Failed:", os.path.basename(f), str(e)[:60])

print("\n🎉 HotSpot PNG generation completed for 2018–2019")
