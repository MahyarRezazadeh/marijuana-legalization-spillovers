# ============================================================
# Build the full Oregon county-year revenue panel: merge the OLCC
# sales totals with county population/distance-to-nonlegal-border,
# and add cumulative dispensary counts (point-in-polygon assigned,
# same convention as the CO/WA revenue-per-store script: open as of
# each year's Dec 31 cutoff, calendar-year reporting).
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
RAW = os.path.join(DATA, 'raw', 'oregon_revenue')
TABLES = os.path.join(BASE, 'tables')

# ---------- Revenue ----------
sales = pd.read_csv(os.path.join(RAW, 'oregon_county_year_sales.csv'))
sales['county_key'] = sales['county'].str.strip().str.upper()

# ---------- Geography (population, distance) ----------
geo = pd.read_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'), dtype={'fips5': str})
geo = geo[geo['state_abbr'] == 'OR'].copy()
geo['county_key'] = geo['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()
geo = geo[['fips5', 'county_key', 'county_name', 'population', 'land_area_sqmi', 'dist_nonlegal_mi']]

panel = sales.merge(geo, on='county_key', how='inner')
print(f"Revenue x geography merge: {len(panel)} of {len(sales)} sales rows matched "
      f"({sales['county_key'].nunique()} sales counties, {geo['county_key'].nunique()} geo counties)")
unmatched = set(sales['county_key']) - set(geo['county_key'])
if unmatched:
    print(f"Unmatched sales counties: {unmatched}")

panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= 70).astype(int)
panel['revenue_per_capita'] = panel['total_sales'] / panel['population']
panel['log_revenue_per_capita'] = np.log(panel['revenue_per_capita'])
panel['log_population'] = np.log(panel['population'])
panel['state_abbr'] = 'OR'
panel = panel.rename(columns={'year': 'fy_year'})

print(f"\nOregon revenue panel: {len(panel)} county-year obs, {panel['fips5'].nunique()} counties, "
      f"years {sorted(panel['fy_year'].unique())}")
print(f"Border (<=70mi): {panel.loc[panel['border_70mi']==1,'county_name'].unique()}")
print(f"Interior: {panel.loc[panel['border_70mi']==0,'fips5'].nunique()} counties")

# ---------- Dispensary counts (point-in-polygon, cumulative-open-by-year) ----------
disp = pd.read_csv(os.path.join(DATA, 'dispensaries_master_geocoded.csv'))
disp = disp[disp['state'] == 'OR'].dropna(subset=['lat', 'lon']).copy()
disp['open_date'] = pd.to_datetime(disp['open_date'], errors='coerce')
print(f"\nOR geocoded dispensaries: {len(disp)} ({disp['open_date'].isna().sum()} missing open_date, treated as always-open)")

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
shp = shp[['GEOID', 'geometry']].rename(columns={'GEOID': 'fips5'})
disp_gdf = gpd.GeoDataFrame(disp, geometry=[Point(xy) for xy in zip(disp['lon'], disp['lat'])], crs='EPSG:4326')
disp_gdf = disp_gdf.to_crs(shp.crs)
joined = gpd.sjoin(disp_gdf, shp, how='left', predicate='within')
unmatched_n = joined['fips5'].isna().sum()
if unmatched_n:
    missing = joined[joined['fips5'].isna()][['uid', 'geometry']]
    nearest = gpd.sjoin_nearest(missing.set_geometry('geometry'), shp, how='left')
    fix = dict(zip(nearest['uid'], nearest['fips5_right'] if 'fips5_right' in nearest.columns else nearest['fips5']))
    joined.loc[joined['fips5'].isna(), 'fips5'] = joined.loc[joined['fips5'].isna(), 'uid'].map(fix)
print(f"Assigned to counties: {joined['fips5'].notna().sum()} of {len(disp)}")

counts = []
for fy in panel['fy_year'].unique():
    cutoff = pd.Timestamp(f'{int(fy)}-12-31')
    open_asof = joined[joined['open_date'].isna() | (joined['open_date'] <= cutoff)]
    c = open_asof.groupby('fips5').size().rename('dispensary_count').reset_index()
    c['fy_year'] = fy
    counts.append(c)
counts = pd.concat(counts, ignore_index=True)

panel = panel.merge(counts, on=['fips5', 'fy_year'], how='left')
panel['dispensary_count'] = panel['dispensary_count'].fillna(0)
panel['revenue_per_store'] = np.where(panel['dispensary_count'] > 0,
                                       panel['total_sales'] / panel['dispensary_count'], np.nan)
panel['log_revenue_per_store'] = np.log(panel['revenue_per_store'])

n_zero_stores = (panel['dispensary_count'] == 0).sum()
print(f"\nCounty-years with zero recorded dispensaries (revenue-per-store undefined): {n_zero_stores}")

panel.to_csv(os.path.join(TABLES, 'oregon_revenue_border_panel.csv'), index=False)
print(f"\nSaved to {os.path.join(TABLES, 'oregon_revenue_border_panel.csv')}")
print(panel[['county_name', 'fy_year', 'border_70mi', 'total_sales', 'population', 'dispensary_count']].head(10))
