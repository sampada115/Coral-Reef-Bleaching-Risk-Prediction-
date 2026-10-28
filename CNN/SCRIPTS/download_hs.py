import os
import requests
from datetime import datetime, timedelta
import calendar
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# ==============================
# REQUEST SESSION (RETRY SAFE)
# ==============================
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
)
session.mount("https://", HTTPAdapter(max_retries=retries))

# ==============================
# LOOP YEARS & MONTHS
# ==============================
for YEAR in [2018, 2019]:
    for MONTH in range(1, 13):

        BASE_URL = (
            "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/"
            f"data/5km/v3.1_op/nc/v1.0/daily/hs/{YEAR}/"
        )

        SAVE_DIR = f"../1_raw_netcdf/hotspot/{YEAR}/{MONTH:02d}"
        os.makedirs(SAVE_DIR, exist_ok=True)

        num_days = calendar.monthrange(YEAR, MONTH)[1]
        start_date = datetime(YEAR, MONTH, 1)
        end_date = datetime(YEAR, MONTH, num_days)

        print(f"\n⬇️ HotSpot {YEAR}-{MONTH:02d}")

        current = start_date
        success = 0
        failed_days = []

        while current <= end_date:
            ymd = current.strftime("%Y%m%d")
            day_str = current.strftime("%d")

            # Remote NOAA filename
            remote_filename = f"ct5km_hs_v3.1_{ymd}.nc"
            url = BASE_URL + remote_filename

            # Local clean filename
            local_filename = f"hotspot_{YEAR}_{MONTH:02d}_{day_str}.nc"
            out_path = os.path.join(SAVE_DIR, local_filename)

            if os.path.exists(out_path):
                current += timedelta(days=1)
                continue

            print(f"📥 {local_filename}", end=" ... ")

            try:
                r = session.get(url, stream=True, timeout=30)

                if r.status_code == 200:
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    print("✅")
                    success += 1
                    time.sleep(0.5)

                else:
                    print(f"❌ HTTP {r.status_code}")
                    failed_days.append(day_str)

            except Exception as e:
                print(f"⚠️ {str(e)[:50]}")
                failed_days.append(day_str)

            current += timedelta(days=1)

        print(f"✅ {success}/{num_days} days downloaded for {YEAR}-{MONTH:02d}")

        if failed_days:
            print(f"⚠️ Failed days: {failed_days}")

print("\n🎉 HotSpot download finished for 2018–2019")
