# ============================================================
# Alcohol placebo, Oregon: aggregate OLCC monthly per-store liquor
# sales to county-calendar-year, both revenue-per-capita and
# revenue-per-store (store count = distinct active liquor_store_number
# in that county-year), matching the two outcomes used for the
# cannabis border design.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'placebo_alcohol')
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

df = pd.read_csv(os.path.join(RAW, 'or_liquor_store_sales_raw.csv'))
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce')
df['county'] = df['county'].str.strip().str.upper().str.replace(' COUNTY', '', regex=False)

# complete-12-months calendar years only, same convention as CO/WA/general-retail builds
counts = df.groupby(['county', 'year'])['month'].nunique().rename('n_months')
annual = df.groupby(['county', 'year']).agg(
    total_sales=('total_sales', 'sum'),
    n_stores=('liquor_store_number', 'nunique')).reset_index()
annual = annual.merge(counts, on=['county', 'year'])
annual = annual[annual['n_months'] == 12].drop(columns='n_months')
annual['county_key'] = annual['county']

geo = pd.read_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'), dtype={'fips5': str})
geo = geo[geo['state_abbr'] == 'OR'].copy()
geo['county_key'] = geo['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()
geo = geo[['fips5', 'county_key', 'county_name', 'population', 'land_area_sqmi', 'dist_nonlegal_mi']]

panel = annual.merge(geo, on='county_key', how='inner')
unmatched = set(annual['county_key']) - set(geo['county_key'])
if unmatched:
    print(f"Unmatched OR counties: {unmatched}")

panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= 70).astype(int)
panel['revenue_per_capita'] = panel['total_sales'] / panel['population']
panel['revenue_per_store'] = panel['total_sales'] / panel['n_stores']
panel['log_revenue_per_capita'] = np.log(panel['revenue_per_capita'])
panel['log_revenue_per_store'] = np.log(panel['revenue_per_store'])
panel['log_population'] = np.log(panel['population'])
panel['state_abbr'] = 'OR'
panel = panel.rename(columns={'year': 'fy_year'})
panel['county_g'] = pd.Categorical(panel['fips5']).codes

print(f"Oregon alcohol panel: {len(panel)} county-year obs, {panel['fips5'].nunique()} counties, "
      f"years {sorted(panel['fy_year'].unique())}")
print(f"Border (<=70mi) counties: {sorted(panel.loc[panel['border_70mi']==1,'county_name'].unique())}")
print(f"Interior counties: {panel.loc[panel['border_70mi']==0,'fips5'].nunique()}")

panel.to_csv(os.path.join(TABLES, 'oregon_alcohol_panel.csv'), index=False)
print(f"\nSaved to {os.path.join(TABLES, 'oregon_alcohol_panel.csv')}")
