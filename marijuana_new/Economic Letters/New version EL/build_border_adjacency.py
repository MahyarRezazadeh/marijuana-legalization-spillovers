# ============================================================
# Border-adjacency robustness check, part 1: for every county in
# the project's never-legal-state universe, compute which STATES
# its own polygon geometrically touches (a literal county-border
# adjacency, in the spirit of Hao & Cowan 2020 / Dube-Lester-Reich,
# which main.tex already cites as the motivation for the 50-mile
# threshold but never implements directly).
#
# This is a ONE-TIME geometric computation (county polygons don't
# move) -- which states border which counties -- kept separate from
# which of those neighboring states is LEGAL in a given year (that
# part varies by transition and is handled downstream in
# build_stacked_border.py using legal_dates_new.csv's retail_date,
# same as every other design in this project).
#
# County shapefile: Census cb_2023_us_county_500k (downloaded fresh
# into data/us_counties_shp/, no auth needed, same source family as
# the state shapefile already used elsewhere in this project).
# State shapefile: reused from the sibling marijuana_housing_overdose
# project (../../marijuana_housing_overdose/data/us_states_shp/), not
# re-downloaded.
# ============================================================
import os
import pandas as pd
import geopandas as gpd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
PARENT_DATA = os.path.join(BASE, '..', 'data')

county_shp = os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp')
state_shp = os.path.join(BASE, '..', '..', '..', 'marijuana_housing_overdose', 'data', 'us_states_shp', 'cb_2023_us_state_500k.shp')

counties = gpd.read_file(county_shp)[['GEOID', 'STATEFP', 'STUSPS', 'NAME', 'geometry']].rename(
    columns={'GEOID': 'fips5', 'STUSPS': 'state_abbr'})
states = gpd.read_file(state_shp)[['STUSPS', 'geometry']].rename(columns={'STUSPS': 'state_abbr'})
# dissolve to one polygon per state (cb_2023 state file is already one row/state, but be safe)
states = states.dissolve(by='state_abbr').reset_index()

# restrict to the project's county universe (31 never-legal-as-of-2020 states),
# same universe as every other design in this project
dist = pd.read_csv(os.path.join(PARENT_DATA, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
counties = counties[counties['fips5'].isin(dist['fips5'])].copy()
print(f"County universe: {len(counties)} (expect 2,331, matching the rest of the project)")

counties = counties.to_crs(states.crs)

# for each never-legal county, find every state whose polygon it touches
# (predicate='touches' -- shares a boundary but does not overlap interiors,
# the correct test for literal border adjacency; a tiny buffer guards
# against sliver gaps from simplified cartographic-boundary geometry)
counties['geom_buffered'] = counties.geometry.buffer(0.001)
touch_rows = []
for _, county in counties.iterrows():
    own_state = county['state_abbr']
    touching = states[states.geometry.intersects(county['geom_buffered']) & (states['state_abbr'] != own_state)]
    for s in touching['state_abbr']:
        touch_rows.append({'fips5': county['fips5'], 'own_state': own_state, 'borders_state': s})

touch_df = pd.DataFrame(touch_rows)
print(f"\nCounties with >=1 cross-state border: {touch_df['fips5'].nunique()} of {len(counties)}")
print(f"Total (county, bordering-state) pairs: {len(touch_df)}")

touch_df.to_csv(os.path.join(DATA, 'county_border_states.csv'), index=False)
print("\nSaved county_border_states.csv")
