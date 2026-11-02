# ============================================================
# Alcohol placebo, Oregon leg: pull OLCC "Distilled Spirits Sales:
# Liquor Store Sales" (data.oregon.gov resource gyqr-vac8) -- monthly,
# per-store, county-tagged. OLCC administers both cannabis and spirits
# retail in OR (control state for spirits), same agency structure that
# made OR's cannabis license data (q32u-cmam) usable earlier in this
# project.
# ============================================================
import os
import time
import requests
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'placebo_alcohol')
os.makedirs(RAW, exist_ok=True)

URL = "https://data.oregon.gov/resource/gyqr-vac8.json"
LIMIT = 5000
rows = []
offset = 0
while True:
    for attempt in range(3):
        try:
            r = requests.get(URL, params={"$limit": LIMIT, "$offset": offset, "$order": ":id"}, timeout=60)
            r.raise_for_status()
            batch = r.json()
            break
        except Exception as e:
            print(f"offset {offset} attempt {attempt+1} failed: {e}")
            time.sleep(2)
    else:
        raise RuntimeError(f"Failed to pull offset {offset} after 3 attempts")
    if not batch:
        break
    rows.extend(batch)
    print(f"pulled {len(batch)} rows at offset {offset} (total {len(rows)})")
    offset += LIMIT
    if len(batch) < LIMIT:
        break

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RAW, 'or_liquor_store_sales_raw.csv'), index=False)
print(f"\nSaved {len(df)} rows to {os.path.join(RAW, 'or_liquor_store_sales_raw.csv')}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Distinct counties: {df['county'].nunique()}")
print(f"Distinct stores: {df['liquor_store_number'].nunique()}")
