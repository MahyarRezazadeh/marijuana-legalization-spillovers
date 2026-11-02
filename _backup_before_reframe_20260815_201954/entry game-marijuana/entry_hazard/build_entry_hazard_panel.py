# ============================================================
# Builds a county-level entry-timing panel to test whether sellers in
# border counties raced to capture cross-border demand faster than
# interior counties, rather than just ending up denser eventually
# (the static count/density test already run in analyze_entry_density.py
# is null; this asks a dynamic question it never could).
#
# For each of the 479 counties in the 9 dispensary-covered states:
#   - days to FIRST dispensary opening after the state's own retail
#     launch date (right-censored at the data cutoff for counties
#     with zero dispensaries as of today)
#   - days from 1st to 2nd dispensary opening, among counties that
#     ever got a first entrant (censored for counties still at 1)
#
# Same point-in-polygon county assignment as build_dispensary_counts.py.
# Same open_date handling convention as the rest of the project: NY's
# ~485 missing-open-date-but-valid-license records cannot be timed and
# are excluded from the entry-TIMING analysis specifically (they were
# correctly included in the static COUNT analysis via lat/lon alone,
# but a missing date cannot be ordered in an event-time model).
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

PROJ_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ_BASE, 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
REF_DATE = pd.Timestamp('2026-08-05')

disp = pd.read_csv(os.path.join(DATA, 'dispensaries_master_geocoded.csv'))
disp = disp.dropna(subset=['lat', 'lon']).copy()
disp['open_date'] = pd.to_datetime(disp['open_date'], errors='coerce')
n_total = len(disp)
n_no_date = disp['open_date'].isna().sum()
disp_timed = disp.dropna(subset=['open_date']).copy()
disp_timed = disp_timed[disp_timed['open_date'] <= REF_DATE].copy()
print(f"Geocoded dispensaries: {n_total}; missing open_date (excluded from timing analysis "
      f"only, incl. NY's ~485 license-valid-but-undated records): {n_no_date}; "
      f"used for entry-timing: {len(disp_timed)}")

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
shp = shp[['GEOID', 'geometry']].rename(columns={'GEOID': 'fips5'})
gdf = gpd.GeoDataFrame(disp_timed, geometry=[Point(xy) for xy in zip(disp_timed['lon'], disp_timed['lat'])],
                        crs='EPSG:4326').to_crs(shp.crs)
joined = gpd.sjoin(gdf, shp, how='left', predicate='within')
unmatched = joined['fips5'].isna().sum()
if unmatched:
    missing = joined[joined['fips5'].isna()][['uid', 'geometry']].set_geometry('geometry')
    nearest = gpd.sjoin_nearest(missing, shp, how='left')
    col = 'fips5_right' if 'fips5_right' in nearest.columns else 'fips5'
    fix = dict(zip(nearest['uid'], nearest[col]))
    joined.loc[joined['fips5'].isna(), 'fips5'] = joined.loc[joined['fips5'].isna(), 'uid'].map(fix)
print(f"Unmatched after nearest-polygon fallback: {joined['fips5'].isna().sum()}")

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
joined = joined.merge(legal[['state_abbr', 'retail_date']],
                       left_on='state', right_on='state_abbr', how='left') \
    if 'state' in joined.columns else joined

# dispensaries_master_geocoded.csv's state field name may differ; recover via county universe instead
universe = pd.read_csv(os.path.join(DATA, 'county_universe_with_distance.csv'), dtype={'fips5': str})
universe['border_70mi'] = (universe['dist_nonlegal_mi'] <= 70).astype(int)
universe = universe.merge(legal[['state_abbr', 'retail_date']], on='state_abbr', how='left')

joined = joined.drop(columns=[c for c in ['retail_date', 'state_abbr'] if c in joined.columns])
joined = joined.merge(universe[['fips5', 'state_abbr', 'retail_date']], on='fips5', how='left')
joined['days_since_legal'] = (joined['open_date'] - joined['retail_date']).dt.days
before_launch = (joined['days_since_legal'] < 0).sum()
print(f"Dispensaries opened before their state's own retail_date (medical-to-retail conversions "
      f"scraped with an early date, or data-entry noise): {before_launch} of {len(joined)} -- "
      f"clipped to 0 rather than dropped (first legal retail day is the earliest meaningful entry time).")
joined['days_since_legal'] = joined['days_since_legal'].clip(lower=0)

entry_dates = joined.groupby('fips5')['days_since_legal'].apply(lambda s: sorted(s.tolist())).rename('entry_days')
panel = universe.merge(entry_dates, on='fips5', how='left')
panel['entry_days'] = panel['entry_days'].apply(lambda x: x if isinstance(x, list) else [])
panel['n_timed_entrants'] = panel['entry_days'].apply(len)
panel['censor_days'] = (REF_DATE - panel['retail_date']).dt.days

# first-entry hazard sample
panel['first_entry_days'] = panel['entry_days'].apply(lambda x: x[0] if len(x) >= 1 else np.nan)
panel['first_entry_event'] = panel['n_timed_entrants'].apply(lambda n: 1 if n >= 1 else 0)
panel['first_entry_time'] = panel.apply(
    lambda r: r['first_entry_days'] if r['first_entry_event'] == 1 else r['censor_days'], axis=1)

# second-entrant hazard sample (time from 1st to 2nd entrant), only defined where >=1 entrant exists
has_first = panel['n_timed_entrants'] >= 1
panel['second_gap_event'] = 0
panel['second_gap_time'] = np.nan
panel.loc[has_first, 'second_gap_event'] = panel.loc[has_first, 'n_timed_entrants'].apply(lambda n: 1 if n >= 2 else 0)
panel.loc[has_first, 'second_gap_time'] = panel.loc[has_first].apply(
    lambda r: (r['entry_days'][1] - r['entry_days'][0]) if r['n_timed_entrants'] >= 2
    else (r['censor_days'] - r['entry_days'][0]), axis=1)
# a handful of counties have censor_days < first_entry_days edge cases (retail_date data noise) -- drop
panel = panel[(panel['first_entry_time'] >= 0)].copy()
if has_first.any():
    bad_gap = panel['second_gap_time'] < 0
    panel.loc[bad_gap, 'second_gap_time'] = 0

panel['log_dist_nonlegal'] = np.log(panel['dist_nonlegal_mi'])
panel['log_population'] = np.log(panel['population'])

panel.drop(columns=['entry_days']).to_csv(os.path.join(OUT, 'entry_hazard_panel.csv'), index=False)
print(f"\nSaved entry_hazard_panel.csv: {len(panel)} counties")
print(f"Counties with >=1 timed entrant: {(panel['first_entry_event']==1).sum()} "
      f"({(panel['first_entry_event']==1).mean()*100:.1f}%)")
print(f"Counties with >=2 timed entrants: {(panel['second_gap_event']==1).sum()}")
print("\nBy state, share with >=1 timed entrant:")
print(panel.groupby('state_abbr')['first_entry_event'].mean().round(3))
