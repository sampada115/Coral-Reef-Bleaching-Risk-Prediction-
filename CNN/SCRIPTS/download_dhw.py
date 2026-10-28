import os
import requests
from calendar import monthrange
import time

BASE_URL = "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/daily/dhw"

for YEAR in [2018, 2019]:
    for MONTH in range(1, 13):

        OUT_DIR = f"../1_raw_netcdf/dhw/{YEAR}/{MONTH:02d}"
        os.makedirs(OUT_DIR, exist_ok=True)

        _, last_day = monthrange(YEAR, MONTH)

        print(f"\n⬇️ DHW {YEAR}-{MONTH:02d} | Days: {last_day}")

        success_count = 0
        failed_days = []

        for day in range(1, last_day + 1):
            date_str = f"{YEAR}{MONTH:02d}{day:02d}"
            filename = f"ct5km_dhw_v3.1_{date_str}.nc"
            outfile = f"{OUT_DIR}/dhw_{YEAR}_{MONTH:02d}_{day:02d}.nc"
            url = f"{BASE_URL}/{YEAR}/{filename}"

            if os.path.exists(outfile):
                continue

            print(f"📥 {date_str}", end=" ... ")

            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()

                with open(outfile, "wb") as f:
                    f.write(r.content)

                print("✅")
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
