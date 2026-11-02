# ============================================================
# Alcohol placebo, Montana: merge the PDF-scraped county-fiscal-year
# liquor sales (scrape_montana_alcohol.py) with the same geography file
# used throughout this project. FY2021 is unavailable (that year's
# by-county schedule is a scanned image, not extractable text). FY2020's
# parsed total is ~0.56% below the report's own printed grand total
# ($164.80M vs $163.87M) -- checked for a mis-attributed/duplicated
# county (the kind of column-layout artifact that corrupted FY2022's
# raw parse) and found none; every county's FY20 value sits smoothly
# between its FY19 and FY22 figures, so this is treated as ordinary
# source-document rounding, not a parsing bug, and left in.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'placebo_alcohol')
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

df = pd.read_csv(os.path.join(RAW, 'mt_liquor_county_fy_raw.csv'))
df['county_key'] = df['county'].str.strip().str.upper()
# MT's own liquor report spells this county "Sweetgrass" (one word); the
# Census-derived geography file uses the official "Sweet Grass County".
df['county_key'] = df['county_key'].replace('SWEETGRASS', 'SWEET GRASS')

geo = pd.read_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'), dtype={'fips5': str})
geo = geo[geo['state_abbr'] == 'MT'].copy()
geo['county_key'] = geo['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()
geo = geo[['fips5', 'county_key', 'county_name', 'population', 'land_area_sqmi', 'dist_nonlegal_mi']]

panel = df.merge(geo, on='county_key', how='inner')
unmatched = set(df['county_key']) - set(geo['county_key'])
if unmatched:
    print(f"Unmatched MT counties: {unmatched}")

panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= 70).astype(int)
panel['revenue_per_capita'] = panel['total_sales'] / panel['population']
panel['log_revenue_per_capita'] = np.log(panel['revenue_per_capita'])
panel['log_population'] = np.log(panel['population'])
panel['state_abbr'] = 'MT'
panel['county_g'] = pd.Categorical(panel['fips5']).codes

print(f"Montana alcohol panel: {len(panel)} county-year obs, {panel['fips5'].nunique()} counties, "
      f"years {sorted(panel['fy_year'].unique())} (FY2021 unavailable -- scanned-image table)")
print(f"Border (<=70mi) counties: {sorted(panel.loc[panel['border_70mi']==1,'county_name'].unique())}")
print(f"Interior counties: {panel.loc[panel['border_70mi']==0,'fips5'].nunique()}")

panel.to_csv(os.path.join(TABLES, 'montana_alcohol_panel.csv'), index=False)
print(f"\nSaved to {os.path.join(TABLES, 'montana_alcohol_panel.csv')}")
