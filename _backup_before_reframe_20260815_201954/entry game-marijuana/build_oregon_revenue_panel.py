# ============================================================
# Pull Oregon's OLCC county-level cannabis sales data (Socrata API,
# data.oregon.gov/resource/5prm-2vmn.json) and build a county-calendar-year
# revenue panel matching the CO/WA convention (total_sales, revenue_per_capita).
#
# Fields in the raw dataset: date, customertype (Consumer/Patient),
# county, producttype, sales, saletype. We sum ALL product types and
# customer types (Consumer+Patient, matching CO/WA's total-sales
# convention which also combines adult-use and medical) to get total
# county revenue per calendar year.
# ============================================================
import os
import time
import requests
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
RAW = os.path.join(DATA, 'raw', 'oregon_revenue')
os.makedirs(RAW, exist_ok=True)

URL = 'https://data.oregon.gov/resource/5prm-2vmn.json'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ---------- Pull all rows via paginated Socrata query ----------
all_rows = []
limit = 50000
offset = 0
while True:
    params = {'$limit': limit, '$offset': offset, '$order': 'date'}
    r = requests.get(URL, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    chunk = r.json()
    if not chunk:
        break
    all_rows.extend(chunk)
    print(f"  pulled {len(chunk)} rows (offset {offset}), total so far {len(all_rows)}")
    offset += limit
    if len(chunk) < limit:
        break
    time.sleep(0.5)

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(RAW, 'oregon_olcc_raw.csv'), index=False)
print(f"\nTotal raw rows: {len(df)}, saved to {RAW}")

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
df['county'] = df['county'].str.strip()

print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"Counties: {df['county'].nunique()}")
print(f"Missing sales values: {df['sales'].isna().sum()} of {len(df)}")

# ---------- Aggregate to county-calendar-year total sales ----------
annual = df.groupby(['county', 'year'])['sales'].sum().reset_index()
annual.columns = ['county', 'year', 'total_sales']

# only keep complete calendar years (12 months of data present for that county-year)
months_present = df.groupby(['county', 'year'])['date'].apply(lambda x: x.dt.month.nunique()).reset_index()
months_present.columns = ['county', 'year', 'n_months']
annual = annual.merge(months_present, on=['county', 'year'])
incomplete = annual[annual['n_months'] < 12]
print(f"\nCounty-years with <12 months of data (dropped): {len(incomplete)}")
if len(incomplete) > 0:
    print(incomplete[['county', 'year', 'n_months']].to_string(index=False))
annual = annual[annual['n_months'] == 12].drop(columns='n_months')

annual.to_csv(os.path.join(RAW, 'oregon_county_year_sales.csv'), index=False)
print(f"\nFinal county-year panel: {len(annual)} obs, {annual['county'].nunique()} counties, "
      f"years {sorted(annual['year'].unique())}")
print(f"Saved to {os.path.join(RAW, 'oregon_county_year_sales.csv')}")
