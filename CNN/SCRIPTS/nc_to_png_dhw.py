import os
import glob
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for YEAR in [2018, 2019]:
    for MONTH in range(1, 13):

        IN_DIR = f"1_raw_netcdf/dhw/{YEAR}/{MONTH:02d}"
        OUT_DIR = f"2_images/dhw/{YEAR}/{MONTH:02d}"
        os.makedirs(OUT_DIR, exist_ok=True)

        files = sorted(glob.glob(f"{IN_DIR}/*.nc"))
        if not files:
            print(f"⚠️ No DHW files for {YEAR}-{MONTH:02d}")
            continue

        print(f"\n🖼️ Converting DHW {YEAR}-{MONTH:02d} | Files: {len(files)}")

        for f in files:
            try:
                ds = xr.open_dataset(f, engine="netcdf4")
            except Exception:
                try:
                    ds = xr.open_dataset(f, engine="scipy")
                except Exception as e:
                    print("❌ Skipping:", os.path.basename(f), e)
                    continue

            dhw = ds["degree_heating_week"]

            if "time" in dhw.dims:
                dhw = dhw.isel(time=0)

            # Crop to GBR
            dhw = dhw.sel(lat=slice(-10, -24), lon=slice(142, 154))
            dhw = dhw.where(dhw >= 0)

            # Extract full date from filename: dhw_YYYY_MM_DD.nc
            base = os.path.basename(f).replace(".nc", "")
            _, y, m, d = base.split("_")

            out = f"{OUT_DIR}/dhw_{y}_{m}_{d}.png"

            plt.figure(figsize=(8, 6))
            dhw.plot(
                cmap="viridis",
                vmin=0,
                vmax=12,
                cbar_kwargs={"label": "Degree Heating Weeks (DHW)"}
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out, dpi=250)
            plt.close()

            print("✅ Saved:", out)

print("\n🎉 DHW PNG generation completed for 2018–2019")
