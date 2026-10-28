import os
import glob
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for YEAR in [2018, 2019]:
    for MONTH in range(1, 13):

        IN_DIR = f"1_raw_netcdf/sst/{YEAR}/{MONTH:02d}"
        OUT_DIR = f"2_images/sst/{YEAR}/{MONTH:02d}"
        os.makedirs(OUT_DIR, exist_ok=True)

        files = sorted(glob.glob(f"{IN_DIR}/*.nc"))
        if not files:
            print(f"⚠️ No SST files for {YEAR}-{MONTH:02d}")
            continue

        print(f"\n🖼️ Converting SST {YEAR}-{MONTH:02d} | Files: {len(files)}")

        for f in files:
            try:
                ds = xr.open_dataset(f)
                sst = ds["analysed_sst"]

                if "time" in sst.dims:
                    sst = sst.isel(time=0)

                # Check lat orientation
                if float(sst.lat[0]) < float(sst.lat[-1]):
                    sst = sst.sel(lat=slice(-24, -10), lon=slice(142, 154))
                else:
                    sst = sst.sel(lat=slice(-10, -24), lon=slice(142, 154))

                if sst.size == 0:
                    continue

                valid_data = sst.where(np.isfinite(sst))
                if valid_data.size == 0 or np.all(np.isnan(valid_data)):
                    continue

                # Convert Kelvin → Celsius if needed
                if float(valid_data.max()) > 100:
                    sst = sst - 273.15

                sst = sst.where(np.isfinite(sst))

                # Extract full date from filename: sst_YYYY_MM_DD.nc
                base = os.path.basename(f).replace(".nc", "")
                _, y, m, d = base.split("_")

                out = f"{OUT_DIR}/sst_{y}_{m}_{d}.png"

                plt.figure(figsize=(8, 6))
                sst.plot(
                    cmap="viridis",
                    vmin=24,
                    vmax=32,
                    cbar_kwargs={"label": "SST (°C)"}
                )
                plt.title(f"SST GBR {y}-{m}-{d}")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.tight_layout()
                plt.savefig(out, dpi=250)
                plt.close()

                print("✅ Saved:", out)

            except Exception as e:
                print("❌ Failed:", os.path.basename(f), str(e)[:60])

print("\n🎉 SST PNG generation completed for 2018–2019")
