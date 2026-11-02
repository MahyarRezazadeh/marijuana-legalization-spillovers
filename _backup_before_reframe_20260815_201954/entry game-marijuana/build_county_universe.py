# ============================================================
# County universe for the 9 dispensary-covered legal states
# (CA, CO, IL, MA, ME, NY, OH, OR, WA), merging:
#   - county centroids (lat/lon, all 3,222 US counties)
#   - 2023 vintage county population estimates (POPESTIMATE2023)
#   - land area (ALAND, from the Census county cartographic shapefile)
# ============================================================
import os
import pandas as pd
import geopandas as gpd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

DISPENSARY_STATES = ['CA', 'CO', 'IL', 'MA', 'ME', 'NY', 'OH', 'OR', 'WA']

centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)

pop = pd.read_csv(os.path.join(DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop = pop[pop['SUMLEV'] == 50].copy()
pop['fips5'] = pop['STATE'].astype(str).str.zfill(2) + pop['COUNTY'].astype(str).str.zfill(3)
pop = pop[['fips5', 'POPESTIMATE2023']].rename(columns={'POPESTIMATE2023': 'population'})

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
land = shp[['GEOID', 'ALAND']].rename(columns={'GEOID': 'fips5', 'ALAND': 'land_area_sqm'})
land['fips5'] = land['fips5'].astype(str).str.zfill(5)
land['land_area_sqmi'] = land['land_area_sqm'] / 2_589_988.11  # sq meters -> sq miles

universe = centroids[centroids['state_abbr'].isin(DISPENSARY_STATES)][
    ['fips5', 'state_abbr', 'county_name', 'lat', 'lon']].copy()
universe = universe.merge(pop, on='fips5', how='left')
universe = universe.merge(land[['fips5', 'land_area_sqmi']], on='fips5', how='left')

print(f"County universe: {len(universe)} counties across {universe['state_abbr'].nunique()} states")
print(universe.groupby('state_abbr').size())
missing_pop = universe['population'].isna().sum()
missing_land = universe['land_area_sqmi'].isna().sum()
print(f"Missing population: {missing_pop}, missing land area: {missing_land}")

universe.to_csv(os.path.join(DATA, 'county_universe.csv'), index=False)
print("\nSaved data/county_universe.csv")
