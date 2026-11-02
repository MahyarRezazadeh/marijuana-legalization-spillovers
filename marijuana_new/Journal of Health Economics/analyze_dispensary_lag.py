# ============================================================
# Rigorous version of the dispensary-opening-lag test: for each
# main-design switcher county, identifies the STATE of its actual
# nearest legal-market county at the moment it switches (not just
# whether a dispensary happens to be nearby), then restricts the
# main-vs-dispensary switch-year comparison to counties whose
# nearest legal state is among the 9 states with dispensary
# geocoding (CA, CO, IL, MA, ME, NY, OH, OR, WA) -- eliminating the
# artifact where "no dispensary switch by 2025" just reflects
# missing dispensary data for an uncovered destination state.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
EL_DATA = os.path.join(BASE, '..', 'Economic Letters', 'data')
DISP_DATA = os.path.join(BASE, '..', 'Economic Letters', 'New version EL', 'data')

EARTH_RADIUS_MI = 3958.8
THRESH = 50.0
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DISPENSARY_STATES = {'CA', 'CO', 'IL', 'MA', 'ME', 'NY', 'OH', 'OR', 'WA'}

# all US counties with lat/lon (need the FULL national county set, not just the
# 31-state never-legal universe, since "legal" counties can be anywhere)
gaz = pd.read_stata(os.path.join(BASE, '..', '..', 'data', 'county_centroids.dta'))
gaz['fips5'] = gaz['fips5'].astype(str).str.zfill(5)
print(f"National county centroids: {len(gaz)}")

legal = pd.read_csv(os.path.join(EL_DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
legal = legal.dropna(subset=['retail_date'])[['state_abbr', 'retail_date']]

def cutoff_date(year):
    return pd.Timestamp(f'{year - 1}-12-31') if year == 2020 else pd.Timestamp(f'{year}-12-31')

def legal_states_asof(year):
    d = cutoff_date(year)
    return set(legal.loc[legal['retail_date'] <= d, 'state_abbr'])

# switcher counties + their switch year (main design, county-distance)
dist = pd.read_csv(os.path.join(EL_DATA, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
dist = dist[dist['dist_2020'] >= THRESH].copy()

def find_switch_year(row):
    for y in YEARS[1:]:
        if row[f'dist_{y-1}'] >= THRESH and row[f'dist_{y}'] < THRESH:
            return y
    return np.nan

dist['switch_year'] = dist.apply(find_switch_year, axis=1)
switchers = dist.dropna(subset=['switch_year']).copy()
switchers['switch_year'] = switchers['switch_year'].astype(int)
print(f"Main-design switcher counties: {len(switchers)}")

# for each switcher, find nearest legal-market county AT ITS OWN SWITCH YEAR, get that county's state
nearest_state = []
for switch_year, grp in switchers.groupby('switch_year'):
    legal_states = legal_states_asof(switch_year)
    legal_counties = gaz[gaz['state_abbr'].isin(legal_states)]
    if legal_counties.empty:
        for _, r in grp.iterrows():
            nearest_state.append({'fips5': r['fips5'], 'nearest_legal_state': None})
        continue
    tree = BallTree(np.radians(legal_counties[['lat', 'lon']].to_numpy()), metric='haversine')
    _, idx = tree.query(np.radians(grp[['lat', 'lon']].to_numpy()), k=1)
    nearest_states = legal_counties.iloc[idx.flatten()]['state_abbr'].values
    for fips, ns in zip(grp['fips5'], nearest_states):
        nearest_state.append({'fips5': fips, 'nearest_legal_state': ns})

ns_df = pd.DataFrame(nearest_state)
switchers = switchers.merge(ns_df, on='fips5', how='left')
switchers['dispensary_covered'] = switchers['nearest_legal_state'].isin(DISPENSARY_STATES)
print(f"\nSwitcher counties whose nearest legal state IS dispensary-covered: "
      f"{switchers['dispensary_covered'].sum()} / {len(switchers)}")
print(switchers['nearest_legal_state'].value_counts())

# now compare against dispensary-design switch years, but ONLY for the covered subset
disp_dist = pd.read_csv(os.path.join(DISP_DATA, 'distance_by_year_dispensary.csv'), dtype={'fips5': str})
disp_dist = disp_dist[disp_dist['dist_2020'] >= THRESH].copy()
disp_dist['switch_year_disp'] = disp_dist.apply(find_switch_year, axis=1)

covered = switchers[switchers['dispensary_covered']].copy()
covered = covered.merge(disp_dist[['fips5', 'switch_year_disp']], on='fips5', how='left')
print(f"\n=== Restricted to {len(covered)} switcher counties whose nearest legal county is in a "
      f"dispensary-covered state ===")
n_no_disp_switch = covered['switch_year_disp'].isna().sum()
print(f"Still never switch under dispensary design by 2025: {n_no_disp_switch} / {len(covered)} "
      f"({n_no_disp_switch/len(covered)*100:.1f}%)")

matched = covered.dropna(subset=['switch_year_disp']).copy()
matched['lag'] = matched['switch_year_disp'] - matched['switch_year']
print(f"\nMatched (both switch): {len(matched)}")
print(matched['lag'].value_counts().sort_index())
print(f"Mean lag: {matched['lag'].mean():.3f}, Median lag: {matched['lag'].median():.1f}")

# market-attractiveness test: does lag correlate with switcher county's own population?
if len(matched) >= 10:
    corr = matched[['lag', 'population']].corr().iloc[0, 1]
    print(f"\nCorrelation(lag, switcher county population): {corr:.3f}")
    print("Population by lag group:")
    print(matched.groupby('lag')['population'].mean())

covered.to_csv(os.path.join(DATA, 'dispensary_lag_covered.csv'), index=False)
print("\nSaved dispensary_lag_covered.csv")
