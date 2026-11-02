# ============================================================
# For every county in the 9 dispensary-covered legal states, compute
# distance to the nearest county in a state where recreational retail
# is NOT yet operating, as of a fixed reference date (2025-12-31 --
# matches the rest of the project's 2020-2025 panel window). This is
# the mirror image of the project's usual "distance to nearest legal
# market" measure: here the county universe is already legal, and we
# want distance to the nearest non-legal (captive-demand) market.
#
# Legal-retail state list uses legal_dates_new.csv's retail_date,
# same convention as the rest of the project (VA excluded despite an
# effective_date -- "no general retail market yet").
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
EARTH_RADIUS_MI = 3958.8
REF_DATE = pd.Timestamp('2025-12-31')

centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
legal_states_asof = set(
    legal.loc[legal['retail_date'].notna() & (legal['retail_date'] <= REF_DATE), 'state_abbr']
)
print(f"Recreational-retail-legal states as of {REF_DATE.date()}: {len(legal_states_asof)}")
print(sorted(legal_states_asof))

nonlegal_counties = centroids[~centroids['state_abbr'].isin(legal_states_asof)].copy()
print(f"\nNon-legal county universe: {len(nonlegal_counties)} counties, "
      f"{nonlegal_counties['state_abbr'].nunique()} states")

universe = pd.read_csv(os.path.join(DATA, 'county_universe.csv'), dtype={'fips5': str})

nonlegal_rad = np.radians(nonlegal_counties[['lat', 'lon']].to_numpy())
tree = BallTree(nonlegal_rad, metric='haversine')

own_rad = np.radians(universe[['lat', 'lon']].to_numpy())
dist_rad, idx = tree.query(own_rad, k=1)
universe['dist_nonlegal_mi'] = dist_rad[:, 0] * EARTH_RADIUS_MI
universe['nearest_nonlegal_state'] = nonlegal_counties.iloc[idx[:, 0]]['state_abbr'].to_numpy()

print(f"\nDistance to nearest non-legal border, by state:")
print(universe.groupby('state_abbr')['dist_nonlegal_mi'].agg(['mean', 'min', 'max']).round(1))

universe.to_csv(os.path.join(DATA, 'county_universe_with_distance.csv'), index=False)
print("\nSaved data/county_universe_with_distance.csv")
