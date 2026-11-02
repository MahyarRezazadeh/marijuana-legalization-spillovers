# ============================================================
# Assign each geocoded dispensary to its containing county via
# point-in-polygon spatial join (more accurate than nearest-centroid
# for counting, since dispensaries near a county boundary could be
# misassigned by centroid distance alone). Aggregate to county-level
# cumulative counts as of the most recent data (max open_date in the
# file) and compute per-capita density.
# ============================================================
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

REF_DATE = pd.Timestamp('2026-08-05')  # "currently open" cutoff -- today

disp = pd.read_csv(os.path.join(DATA, 'dispensaries_master_geocoded.csv'))
disp = disp.dropna(subset=['lat', 'lon']).copy()
disp['open_date'] = pd.to_datetime(disp['open_date'], errors='coerce')

# Count as "currently open" if open_date is missing (485 NY records have
# valid license numbers/geocoding but no scraped open date -- real
# licensed stores, not bad data) OR open_date <= today. Only exclude
# records with a KNOWN future open_date (14 OR licenses not yet open).
n_before = len(disp)
n_future = (disp['open_date'] > REF_DATE).sum()
n_missing_date = disp['open_date'].isna().sum()
disp = disp[disp['open_date'].isna() | (disp['open_date'] <= REF_DATE)].copy()
print(f"Geocoded dispensaries: {n_before}; missing open_date (kept, treated as already-open): "
      f"{n_missing_date}; excluded as not-yet-open: {n_future}; used: {len(disp)}")

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
shp = shp[['GEOID', 'geometry']].rename(columns={'GEOID': 'fips5'})

disp_gdf = gpd.GeoDataFrame(
    disp, geometry=[Point(xy) for xy in zip(disp['lon'], disp['lat'])], crs='EPSG:4326')
disp_gdf = disp_gdf.to_crs(shp.crs)

joined = gpd.sjoin(disp_gdf, shp, how='left', predicate='within')
unmatched = joined['fips5'].isna().sum()
print(f"Dispensaries unmatched to any county polygon: {unmatched} (of {len(joined)})")

# a handful of points can fall just outside a polygon due to
# cartographic simplification; fall back to nearest polygon for those
if unmatched:
    missing = joined[joined['fips5'].isna()][['uid', 'geometry']]
    nearest = gpd.sjoin_nearest(missing.set_geometry('geometry'), shp, how='left')
    fix = dict(zip(nearest['uid'], nearest['fips5_right'] if 'fips5_right' in nearest.columns else nearest['fips5']))
    joined.loc[joined['fips5'].isna(), 'fips5'] = joined.loc[joined['fips5'].isna(), 'uid'].map(fix)
    print(f"  Fixed via nearest-polygon fallback: {joined['fips5'].notna().sum()} now matched")

counts = joined.groupby('fips5').size().rename('dispensary_count').reset_index()

universe = pd.read_csv(os.path.join(DATA, 'county_universe_with_distance.csv'), dtype={'fips5': str})
universe = universe.merge(counts, on='fips5', how='left')
universe['dispensary_count'] = universe['dispensary_count'].fillna(0).astype(int)
universe['dispensary_per_1000'] = universe['dispensary_count'] / universe['population'] * 1000
universe['dispensary_per_sqmi100'] = universe['dispensary_count'] / universe['land_area_sqmi'] * 100

print(f"\nTotal dispensaries assigned into county universe: {universe['dispensary_count'].sum()} of {len(disp)}")
print(f"Counties with zero dispensaries: {(universe['dispensary_count'] == 0).sum()} of {len(universe)}")
print("\nDispensary density by state:")
print(universe.groupby('state_abbr').agg(
    counties=('fips5', 'size'),
    total_disp=('dispensary_count', 'sum'),
    mean_per_1000=('dispensary_per_1000', 'mean'),
).round(3))

universe.to_csv(os.path.join(DATA, 'county_panel_entry.csv'), index=False)
print("\nSaved data/county_panel_entry.csv")
