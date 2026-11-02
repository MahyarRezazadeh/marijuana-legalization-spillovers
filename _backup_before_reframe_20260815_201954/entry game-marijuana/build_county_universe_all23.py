# ============================================================
# County universe for ALL 23 states with an operating recreational
# retail market as of 2026-08-05 (extends build_county_universe.py's
# original 9-dispensary-covered-state scope to the full current legal
# universe, since the underlying sources -- centroids, population,
# land area -- are already national).
# ============================================================
import os
import pandas as pd
import geopandas as gpd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
REF_DATE = pd.Timestamp('2026-08-05')

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
LEGAL_STATES = sorted(legal.loc[legal['retail_date'].notna() & (legal['retail_date'] <= REF_DATE), 'state_abbr'])
print(f"Legal-retail states as of {REF_DATE.date()}: {len(LEGAL_STATES)}")
print(LEGAL_STATES)

centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)

pop = pd.read_csv(os.path.join(DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop = pop[pop['SUMLEV'] == 50].copy()
pop['fips5'] = pop['STATE'].astype(str).str.zfill(2) + pop['COUNTY'].astype(str).str.zfill(3)
pop = pop[['fips5', 'POPESTIMATE2023']].rename(columns={'POPESTIMATE2023': 'population'})

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
land = shp[['GEOID', 'ALAND']].rename(columns={'GEOID': 'fips5', 'ALAND': 'land_area_sqm'})
land['fips5'] = land['fips5'].astype(str).str.zfill(5)
land['land_area_sqmi'] = land['land_area_sqm'] / 2_589_988.11

universe = centroids[centroids['state_abbr'].isin(LEGAL_STATES)][
    ['fips5', 'state_abbr', 'county_name', 'lat', 'lon']].copy()
universe = universe.merge(pop, on='fips5', how='left')
universe = universe.merge(land[['fips5', 'land_area_sqmi']], on='fips5', how='left')

print(f"\nCounty universe: {len(universe)} counties across {universe['state_abbr'].nunique()} states")
print(universe.groupby('state_abbr').size())
print(f"Missing population: {universe['population'].isna().sum()}, missing land area: {universe['land_area_sqmi'].isna().sum()}")

universe.to_csv(os.path.join(DATA, 'county_universe_all23.csv'), index=False)
print("\nSaved data/county_universe_all23.csv")
