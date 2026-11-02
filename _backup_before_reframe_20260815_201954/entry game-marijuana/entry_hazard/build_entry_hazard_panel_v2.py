# ============================================================
# Extends build_entry_hazard_panel.py (which only kept 1st-entrant and
# 1st-to-2nd-entrant timing) in two ways needed to properly characterize
# the "second entrant" result before writing anything up:
#
# (1) Preserves the FULL sorted entry-date sequence per county so we can
#     also build 2nd-to-3rd and 3rd-to-4th entrant gap hazard samples --
#     needed to tell whether the speed effect found for the 2nd entrant
#     is specific to entrant #2, or just a generic "more competition
#     nearby -> faster entry at every rank" pattern.
#
# (2) Merges in the captive non-legal-population-within-70mi measure
#     already used for the revenue-per-store mechanism check
#     (analyze_revenue_perstore_mechanism.py), extended here to the full
#     9-state/479-county entry-density universe rather than just CO/WA,
#     so we can test whether the 2nd-entrant hazard effect is
#     concentrated in counties actually exposed to a large cross-border
#     demand pool, not just "close to some non-legal county" (a distance
#     measure that returns the nearest county's distance regardless of
#     that county's population, e.g. a tiny rural county vs. a big metro
#     both count as "border" if they are the nearest).
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree

PROJ_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ_BASE, 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
REF_DATE = pd.Timestamp('2026-08-05')
EARTH_RADIUS_MI = 3958.8

disp = pd.read_csv(os.path.join(DATA, 'dispensaries_master_geocoded.csv'))
disp = disp.dropna(subset=['lat', 'lon']).copy()
disp['open_date'] = pd.to_datetime(disp['open_date'], errors='coerce')
disp_timed = disp.dropna(subset=['open_date']).copy()
disp_timed = disp_timed[disp_timed['open_date'] <= REF_DATE].copy()

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
shp = shp[['GEOID', 'geometry']].rename(columns={'GEOID': 'fips5'})
gdf = gpd.GeoDataFrame(disp_timed, geometry=[Point(xy) for xy in zip(disp_timed['lon'], disp_timed['lat'])],
                        crs='EPSG:4326').to_crs(shp.crs)
joined = gpd.sjoin(gdf, shp, how='left', predicate='within')
if joined['fips5'].isna().any():
    missing = joined[joined['fips5'].isna()][['uid', 'geometry']].set_geometry('geometry')
    nearest = gpd.sjoin_nearest(missing, shp, how='left')
    col = 'fips5_right' if 'fips5_right' in nearest.columns else 'fips5'
    fix = dict(zip(nearest['uid'], nearest[col]))
    joined.loc[joined['fips5'].isna(), 'fips5'] = joined.loc[joined['fips5'].isna(), 'uid'].map(fix)

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
universe = pd.read_csv(os.path.join(DATA, 'county_universe_with_distance.csv'), dtype={'fips5': str})
universe['border_70mi'] = (universe['dist_nonlegal_mi'] <= 70).astype(int)
universe = universe.merge(legal[['state_abbr', 'retail_date']], on='state_abbr', how='left')

joined = joined.drop(columns=[c for c in ['retail_date', 'state_abbr'] if c in joined.columns])
joined = joined.merge(universe[['fips5', 'state_abbr', 'retail_date']], on='fips5', how='left')
joined['days_since_legal'] = (joined['open_date'] - joined['retail_date']).dt.days
joined['days_since_legal'] = joined['days_since_legal'].clip(lower=0)

entry_dates = joined.groupby('fips5')['days_since_legal'].apply(lambda s: sorted(s.tolist())).rename('entry_days')
panel = universe.merge(entry_dates, on='fips5', how='left')
panel['entry_days'] = panel['entry_days'].apply(lambda x: x if isinstance(x, list) else [])
panel['n_timed_entrants'] = panel['entry_days'].apply(len)
panel['censor_days'] = (REF_DATE - panel['retail_date']).dt.days

panel['first_entry_days'] = panel['entry_days'].apply(lambda x: x[0] if len(x) >= 1 else np.nan)
panel['first_entry_event'] = panel['n_timed_entrants'].apply(lambda n: 1 if n >= 1 else 0)
panel['first_entry_time'] = panel.apply(
    lambda r: r['first_entry_days'] if r['first_entry_event'] == 1 else r['censor_days'], axis=1)


def gap_sample(panel, from_rank, to_rank):
    """Rows where the county reached `from_rank` entrants: event=1 and gap
    time = entry_days[to_rank-1]-entry_days[from_rank-1] if the county also
    reached `to_rank`; else event=0, gap time = censor_days-entry_days[from_rank-1]."""
    has_from = panel['n_timed_entrants'] >= from_rank
    event = np.where(panel.loc[has_from, 'n_timed_entrants'] >= to_rank, 1, 0)
    time = panel.loc[has_from].apply(
        lambda r: (r['entry_days'][to_rank - 1] - r['entry_days'][from_rank - 1])
        if r['n_timed_entrants'] >= to_rank
        else (r['censor_days'] - r['entry_days'][from_rank - 1]), axis=1)
    out = panel.loc[has_from].copy()
    out[f'gap_{from_rank}to{to_rank}_event'] = event
    out[f'gap_{from_rank}to{to_rank}_time'] = time.clip(lower=0)
    return out

gap12 = gap_sample(panel, 1, 2).rename(columns={'gap_1to2_event': 'second_gap_event', 'gap_1to2_time': 'second_gap_time'})
gap23 = gap_sample(panel, 2, 3)
gap34 = gap_sample(panel, 3, 4)

panel = panel.merge(gap12[['fips5', 'second_gap_event', 'second_gap_time']], on='fips5', how='left')
panel = panel.merge(gap23[['fips5', 'gap_2to3_event', 'gap_2to3_time']], on='fips5', how='left')
panel = panel.merge(gap34[['fips5', 'gap_3to4_event', 'gap_3to4_time']], on='fips5', how='left')

panel['log_dist_nonlegal'] = np.log(panel['dist_nonlegal_mi'])
panel['log_population'] = np.log(panel['population'])

# ---------- captive non-legal population within 70mi, full 9-state universe ----------
legal_states_asof = set(legal.loc[legal['retail_date'].notna() & (legal['retail_date'] <= REF_DATE), 'state_abbr'])
centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)
pop = pd.read_csv(os.path.join(DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop = pop[pop['SUMLEV'] == 50].copy()
pop['fips5'] = pop['STATE'].astype(str).str.zfill(2) + pop['COUNTY'].astype(str).str.zfill(3)
pop = pop[['fips5', 'POPESTIMATE2023']].rename(columns={'POPESTIMATE2023': 'nl_population'})
centroids = centroids.merge(pop, on='fips5', how='left')
nonlegal = centroids[~centroids['state_abbr'].isin(legal_states_asof)].dropna(subset=['nl_population']).copy()

nonlegal_rad = np.radians(nonlegal[['lat', 'lon']].to_numpy())
tree = BallTree(nonlegal_rad, metric='haversine')
radius_rad = 70 / EARTH_RADIUS_MI
own = panel[['fips5']].merge(centroids[['fips5', 'lat', 'lon']], on='fips5', how='left')
own_rad = np.radians(own[['lat', 'lon']].to_numpy())
indices = tree.query_radius(own_rad, r=radius_rad)
captive_pop = [nonlegal.iloc[idx]['nl_population'].sum() for idx in indices]
panel['captive_nonlegal_pop_70mi'] = captive_pop
panel['log_captive_pop'] = np.log(panel['captive_nonlegal_pop_70mi'] + 1)
panel['has_captive_pop'] = (panel['captive_nonlegal_pop_70mi'] > 0).astype(int)

print(f"Counties with any captive nonlegal population within 70mi: {panel['has_captive_pop'].sum()} of {len(panel)}")
print(f"By state:\n{panel.groupby('state_abbr')['has_captive_pop'].sum()}")

panel.drop(columns=['entry_days']).to_csv(os.path.join(OUT, 'entry_hazard_panel_v2.csv'), index=False)
print(f"\nSaved entry_hazard_panel_v2.csv: {len(panel)} counties")
print(f"1st entrant: {panel['first_entry_event'].sum()} events")
print(f"1->2 gap: {panel['second_gap_event'].notna().sum()} at risk, {panel['second_gap_event'].sum()} events")
print(f"2->3 gap: {panel['gap_2to3_event'].notna().sum()} at risk, {panel['gap_2to3_event'].sum()} events")
print(f"3->4 gap: {panel['gap_3to4_event'].notna().sum()} at risk, {panel['gap_3to4_event'].sum()} events")
