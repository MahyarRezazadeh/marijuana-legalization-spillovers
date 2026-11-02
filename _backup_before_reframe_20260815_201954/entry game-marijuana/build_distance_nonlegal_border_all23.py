# ============================================================
# Distance to nearest non-legal-state county, for all 991 counties in
# the 23 currently-legal-retail states, as of 2026-08-05. Extends
# build_distance_nonlegal_border.py (originally 9 dispensary-covered
# states, 50mi threshold) to the full current legal universe.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
EARTH_RADIUS_MI = 3958.8
REF_DATE = pd.Timestamp('2026-08-05')

centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
legal_states_asof = set(legal.loc[legal['retail_date'].notna() & (legal['retail_date'] <= REF_DATE), 'state_abbr'])

nonlegal_counties = centroids[~centroids['state_abbr'].isin(legal_states_asof)].copy()
print(f"Non-legal county universe: {len(nonlegal_counties)} counties, {nonlegal_counties['state_abbr'].nunique()} states")

universe = pd.read_csv(os.path.join(DATA, 'county_universe_all23.csv'), dtype={'fips5': str})

nonlegal_rad = np.radians(nonlegal_counties[['lat', 'lon']].to_numpy())
tree = BallTree(nonlegal_rad, metric='haversine')
own_rad = np.radians(universe[['lat', 'lon']].to_numpy())
dist_rad, idx = tree.query(own_rad, k=1)
universe['dist_nonlegal_mi'] = dist_rad[:, 0] * EARTH_RADIUS_MI
universe['nearest_nonlegal_state'] = nonlegal_counties.iloc[idx[:, 0]]['state_abbr'].to_numpy()

print("\nDistance to nearest non-legal border, by state (min/mean/max, miles):")
summary = universe.groupby('state_abbr')['dist_nonlegal_mi'].agg(['min', 'mean', 'max']).round(1).sort_values('min')
print(summary.to_string())

universe.to_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'), index=False)
print("\nSaved data/county_universe_all23_with_distance.csv")

print("\nStates with NO county within 70mi of a non-legal border (weak/no border variation):")
weak = summary[summary['min'] > 70]
print(weak.index.tolist() if len(weak) else "none")
