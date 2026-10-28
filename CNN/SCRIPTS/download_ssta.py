import os
import requests
from calendar import monthrange
import time

for YEAR in [2018, 2019]:
    for MONTH in range(1, 13):

        OUT_DIR = f"../1_raw_netcdf/sst_anomaly/{YEAR}/{MONTH:02d}"
        os.makedirs(OUT_DIR, exist_ok=True)

        _, last_day = monthrange(YEAR, MONTH)

        print(f"\n⬇️ SST Anomaly {YEAR}-{MONTH:02d}")
        print(f"📅 Days in month: {last_day}")

        success_count = 0
        failed_days = []

        for day in range(1, last_day + 1):
            outfile = f"{OUT_DIR}/sst_anomaly_{YEAR}_{MONTH:02d}_{day:02d}.nc"

            # Skip if already downloaded
            if os.path.exists(outfile):
                continue

            url = (
                "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.nc"
                "?CRW_SSTANOMALY"
                f"[({YEAR}-{MONTH:02d}-{day:02d}T00:00:00Z)]"
                "[(-24):1:(-10)]"
                "[(142):1:(154)]"
            )

            print(f"📥 {YEAR}-{MONTH:02d}-{day:02d}", end=" ... ")

            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()

                with open(outfile, "wb") as f:
                    f.write(r.content)

                file_size_kb = len(r.content) / 1024
                print(f"✅ ({file_size_kb:.2f} KB)")
                success_count += 1
                time.sleep(0.5)

            except requests.exceptions.HTTPError:
                print(f"❌ HTTP {r.status_code}")
                failed_days.append(day)

            except requests.exceptions.Timeout:
                print("❌ Timeout")
                failed_days.append(day)

            except Exception as e:
                print(f"❌ {str(e)[:50]}")
                failed_days.append(day)

        print(f"✅ {success_count}/{last_day} days downloaded for {YEAR}-{MONTH:02d}")

        if failed_days:
            print(f"⚠️ Failed days: {failed_days}")

print("\n🎉 SST Anomaly download completed for 2018–2019")
