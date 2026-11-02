# ============================================================
# Retry the one failed month (2022-12) and build the Montana
# county-calendar-year revenue panel from the scraped monthly data,
# deduping the two months (2024-01, 2024-04) whose pages rendered the
# table twice (verified identical values, harmless double-render).
# Confidentiality-suppressed county-months are NaN, not zero -- a
# county-year is only kept if all 12 months are disclosed.
# ============================================================
import os
import time
import requests
import pandas as pd
from io import StringIO

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'montana_revenue')
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

df = pd.read_csv(os.path.join(RAW, 'montana_county_month_sales.csv'))
df = df.drop_duplicates(subset=['county', 'year', 'month'])
print(f"After dedup: {len(df)} county-month rows")

# retry December 2022
if not ((df['year'] == 2022) & (df['month'] == 12)).any():
    url = "https://revenue.mt.gov/card/cannabis/sales-reports/2022-12-county-sales"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                tables = pd.read_html(StringIO(r.text))
                tbl = next(t for t in tables if any('County' in str(c) for c in t.columns))
                total_col = [c for c in tbl.columns if 'Estimated Total County Sales' in str(c)][0]
                sub = tbl[['County', total_col]].copy()
                sub.columns = ['county', 'sales_raw']
                sub['sales'] = pd.to_numeric(sub['sales_raw'].astype(str).str.replace(r'[$,]', '', regex=True),
                                              errors='coerce')
                sub = sub.dropna(subset=['county'])
                sub = sub[~sub['county'].astype(str).str.contains('Total', case=False, na=False)]
                sub['year'] = 2022
                sub['month'] = 12
                df = pd.concat([df, sub[['county', 'year', 'month', 'sales']]], ignore_index=True)
                print(f"2022-12: retry OK, {sub['sales'].notna().sum()} of {len(sub)} disclosed")
                break
        except Exception as e:
            print(f"2022-12 retry {attempt+1} failed: {e}")
            time.sleep(2)

df.to_csv(os.path.join(RAW, 'montana_county_month_sales_clean.csv'), index=False)

# ---------- Aggregate to county-calendar-year ----------
annual = df.groupby(['county', 'year']).agg(
    total_sales=('sales', 'sum'), n_months=('sales', 'count')).reset_index()
complete = annual[annual['n_months'] == 12].drop(columns='n_months')
incomplete = annual[annual['n_months'] < 12]
print(f"\nComplete county-years (12/12 months disclosed): {len(complete)}")
print(f"Incomplete county-years (dropped, some months suppressed): {len(incomplete)}")

# ---------- Merge with geography ----------
geo = pd.read_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'), dtype={'fips5': str})
geo = geo[geo['state_abbr'] == 'MT'].copy()
geo['county_key'] = geo['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()
geo = geo[['fips5', 'county_key', 'county_name', 'population', 'land_area_sqmi', 'dist_nonlegal_mi']]

complete['county_key'] = complete['county'].str.strip().str.upper()
panel = complete.merge(geo, on='county_key', how='inner')
print(f"\nRevenue x geography merge: {len(panel)} of {len(complete)} matched")
unmatched = set(complete['county_key']) - set(geo['county_key'])
if unmatched:
    print(f"Unmatched: {unmatched}")

import numpy as np
panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= 70).astype(int)
panel['revenue_per_capita'] = panel['total_sales'] / panel['population']
n_zero = (panel['revenue_per_capita'] <= 0).sum()
print(f"\nCounty-years with zero recorded sales (no licensed retailer -- dropped, log undefined): {n_zero}")
panel = panel[panel['revenue_per_capita'] > 0].copy()
panel['log_revenue_per_capita'] = np.log(panel['revenue_per_capita'])
panel['log_population'] = np.log(panel['population'])
panel['state_abbr'] = 'MT'
panel = panel.rename(columns={'year': 'fy_year'})
panel['county_g'] = pd.Categorical(panel['fips5']).codes

print(f"\nMontana revenue panel: {len(panel)} county-year obs, {panel['fips5'].nunique()} counties, "
      f"years {sorted(panel['fy_year'].unique())}")
print(f"Border (<=70mi) counties: {sorted(panel.loc[panel['border_70mi']==1,'county_name'].unique())}")
print(f"Interior counties: {panel.loc[panel['border_70mi']==0,'fips5'].nunique()}")

panel.to_csv(os.path.join(TABLES, 'montana_revenue_percapita_panel.csv'), index=False)
print(f"\nSaved to {os.path.join(TABLES, 'montana_revenue_percapita_panel.csv')}")
