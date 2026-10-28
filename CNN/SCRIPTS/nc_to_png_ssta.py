import os
import glob
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for YEAR in [2018, 2019]:
    for MONTH in range(1, 13):

        IN_DIR = f"1_raw_netcdf/sst_anomaly/{YEAR}/{MONTH:02d}"
        OUT_DIR = f"2_images/sst_anomaly/{YEAR}/{MONTH:02d}"
        os.makedirs(OUT_DIR, exist_ok=True)

        files = sorted(glob.glob(f"{IN_DIR}/*.nc"))
        if not files:
            continue

        print(f"\n🖼️ Converting SSTA {YEAR}-{MONTH:02d} | Files: {len(files)}")

        for f in files:
            try:
                ds = xr.open_dataset(f)
                ssta = ds["CRW_SSTANOMALY"]

                if "time" in ssta.dims:
                    ssta = ssta.isel(time=0)

                # Crop GBR
                ssta = ssta.sel(
                    latitude=slice(-10, -24),
                    longitude=slice(142, 154)
                )

                ssta = ssta.where(np.isfinite(ssta))

                # Extract day from filename: sst_anomaly_YYYY_MM_DD.nc
                day = os.path.basename(f).split("_")[-1].replace(".nc", "")

                out = f"{OUT_DIR}/ssta_{YEAR}_{MONTH:02d}_{day}.png"

                plt.figure(figsize=(8, 6))
                ssta.plot(
                    cmap="RdBu_r",
                    vmin=-3,
                    vmax=3,
                    cbar_kwargs={"label": "SST Anomaly (°C)"}
                )
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out, dpi=250)
                plt.close()

                print("✅ Saved:", out)

            except Exception as e:
                print("❌ Failed:", os.path.basename(f), str(e)[:60])

print("\n🎉 SST Anomaly PNG generation completed for 2018–2019")
