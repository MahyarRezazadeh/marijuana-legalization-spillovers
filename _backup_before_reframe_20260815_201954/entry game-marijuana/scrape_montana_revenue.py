# ============================================================
# Scrape Montana's monthly per-county cannabis sales reports
# (revenue.mt.gov/card/cannabis/sales-reports/YYYY-MM-county-sales).
# No bulk CSV/API exists for this data -- each month is a separate
# static HTML page with one table (County, tax rates, adult-use sales,
# medical sales, total county sales). Montana retail launched
# 2022-01-01, so we try every month from Jan 2022 through the present.
# Small counties are frequently suppressed for confidentiality (fewer
# than ~3 licensed retailers) -- these show as "Not disclosed" and are
# recorded as missing, not zero.
# ============================================================
import os
import re
import time
import requests
import pandas as pd
from datetime import date

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'montana_revenue')
os.makedirs(RAW, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

months = []
y, m = 2022, 1
today = date.today()
while (y, m) <= (today.year, today.month):
    months.append((y, m))
    m += 1
    if m > 12:
        m = 1
        y += 1

rows = []
ok, missing = 0, 0
for y, m in months:
    url = f"https://revenue.mt.gov/card/cannabis/sales-reports/{y}-{m:02d}-county-sales"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
    except Exception as e:
        print(f"{y}-{m:02d}: request error {e}")
        missing += 1
        continue
    if r.status_code != 200:
        print(f"{y}-{m:02d}: HTTP {r.status_code} (not yet published)")
        missing += 1
        continue
    try:
        tables = pd.read_html(r.text)
    except ValueError:
        print(f"{y}-{m:02d}: no table found")
        missing += 1
        continue
    # find the table with a 'County' column
    tbl = None
    for t in tables:
        if any('County' in str(c) for c in t.columns):
            tbl = t
            break
    if tbl is None:
        print(f"{y}-{m:02d}: table found but no County column")
        missing += 1
        continue

    total_col = [c for c in tbl.columns if 'Estimated Total County Sales' in str(c)]
    if not total_col:
        print(f"{y}-{m:02d}: no total-sales column")
        missing += 1
        continue
    total_col = total_col[0]

    sub = tbl[['County', total_col]].copy()
    sub.columns = ['county', 'sales_raw']
    sub['sales'] = sub['sales_raw'].astype(str).str.replace(r'[$,]', '', regex=True)
    sub['sales'] = pd.to_numeric(sub['sales'], errors='coerce')  # 'Not disclosed'/'*' -> NaN
    sub = sub.dropna(subset=['county'])
    sub = sub[~sub['county'].astype(str).str.contains('Total', case=False, na=False)]
    sub['year'] = y
    sub['month'] = m
    rows.append(sub[['county', 'year', 'month', 'sales']])
    ok += 1
    print(f"{y}-{m:02d}: OK, {sub['sales'].notna().sum()} of {len(sub)} counties disclosed")
    time.sleep(0.3)

print(f"\nMonths pulled: {ok} ok, {missing} missing/unavailable")
panel = pd.concat(rows, ignore_index=True)
panel.to_csv(os.path.join(RAW, 'montana_county_month_sales.csv'), index=False)
print(f"Saved {len(panel)} county-month rows to {os.path.join(RAW, 'montana_county_month_sales.csv')}")
