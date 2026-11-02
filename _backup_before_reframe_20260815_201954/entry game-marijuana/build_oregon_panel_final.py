# ============================================================
# Final Oregon panel build. Revenue-per-capita: full county-year panel
# (OLCC Socrata sales data, 2017-2025, matched to population/distance).
#
# Revenue-per-store: the geocoded dispensaries_master_geocoded.csv (OR
# subset) turned out to be unreliable for this purpose -- several
# records geocode to counties in OTHER states entirely (King, Tyler,
# Mecosta, McKean, Whiteside, Otero, Gibson -- none are Oregon
# counties), and Baker County (one of the 3 border counties) has ZERO
# matched dispensaries despite having real recorded revenue in every
# year 2017-2025. We instead pull retailer COUNTS directly from OLCC's
# own license dataset (already cited as OLCCLicenses), which tags
# county natively -- no geocoding needed. But that dataset's
# "effective_date" reflects each license's latest renewal, not its
# original opening date, so it cannot support a historical panel --
# only a current snapshot of active licenses. We therefore build
# revenue-per-store as a single cross-section: 2025 revenue (last
# complete calendar year) against currently-active retailer counts,
# the same limitation and approach already used for the CO general
# retail placebo's per-retailer check.
# ============================================================
import os
import time
import requests
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
RAW = os.path.join(DATA, 'raw', 'oregon_revenue')
TABLES = os.path.join(BASE, 'tables')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ---------- Revenue panel (per-capita, all years) ----------
sales = pd.read_csv(os.path.join(RAW, 'oregon_county_year_sales.csv'))
sales['county_key'] = sales['county'].str.strip().str.upper()

geo = pd.read_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'), dtype={'fips5': str})
geo = geo[geo['state_abbr'] == 'OR'].copy()
geo['county_key'] = geo['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()
geo = geo[['fips5', 'county_key', 'county_name', 'population', 'land_area_sqmi', 'dist_nonlegal_mi']]

panel = sales.merge(geo, on='county_key', how='inner')
panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= 70).astype(int)
panel['revenue_per_capita'] = panel['total_sales'] / panel['population']
panel['log_revenue_per_capita'] = np.log(panel['revenue_per_capita'])
panel['log_population'] = np.log(panel['population'])
panel['state_abbr'] = 'OR'
panel = panel.rename(columns={'year': 'fy_year'})
panel['county_g'] = pd.Categorical(panel['fips5']).codes

print(f"Oregon revenue-per-capita panel: {len(panel)} county-year obs, {panel['fips5'].nunique()} counties, "
      f"years {sorted(panel['fy_year'].unique())}")
print(f"Border counties: {sorted(panel.loc[panel['border_70mi']==1,'county_name'].unique())}")
panel.to_csv(os.path.join(TABLES, 'oregon_revenue_percapita_panel.csv'), index=False)

# ---------- Retailer counts: current snapshot from OLCC license API ----------
url = 'https://data.oregon.gov/resource/q32u-cmam.json'
params = {'license_type': 'RECREATIONAL RETAILER', 'license_expired': 'No',
          '$select': 'county,count(*) as n', '$group': 'county', '$limit': 100}
r = requests.get(url, params=params, headers=HEADERS, timeout=30)
r.raise_for_status()
retailers = pd.DataFrame(r.json())
retailers['n'] = retailers['n'].astype(int)
retailers['county_key'] = retailers['county'].str.strip().str.upper()
retailers.to_csv(os.path.join(RAW, 'oregon_active_retailers_by_county.csv'), index=False)
print(f"\nActive retailer counts pulled for {len(retailers)} counties (current snapshot)")

cross = panel[panel['fy_year'] == 2025].merge(
    retailers[['county_key', 'n']], on='county_key', how='left')
cross['dispensary_count'] = cross['n'].fillna(0)
cross = cross[cross['dispensary_count'] > 0].copy()
cross['revenue_per_store'] = cross['total_sales'] / cross['dispensary_count']
cross['log_revenue_per_store'] = np.log(cross['revenue_per_store'])

print(f"\nRevenue-per-store cross-section (2025 revenue / current active retailers): "
      f"{len(cross)} counties, border={cross['border_70mi'].sum()}")
print(cross[['county_name', 'border_70mi', 'total_sales', 'dispensary_count', 'revenue_per_store']]
      .sort_values('border_70mi', ascending=False).to_string(index=False))

cross.to_csv(os.path.join(TABLES, 'oregon_revenue_perstore_cross2025.csv'), index=False)
print(f"\nSaved panel and cross-section to {TABLES}")
