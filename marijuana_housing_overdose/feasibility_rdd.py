# ============================================================
# FEASIBILITY CHECK for a spatial RDD / diff-in-discontinuities
# design: running variable = signed distance to the ACTUAL state
# border line (never-legal state = negative, legal-market state
# = positive), not distance to nearest legal county centroid.
# Checks, for a range of candidate bandwidths, how many counties
# on each side have non-suppressed overdose data in BOTH 2020
# and 2023 (needed for the diff-in-disc estimator). No regression
# is run here -- just a sample-size/power scoping exercise.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

LEGAL_STATES = {
    'WA': 2014, 'CO': 2014, 'OR': 2015, 'AK': 2016, 'NV': 2017, 'MA': 2018,
    'CA': 2018, 'MI': 2019, 'IL': 2020, 'AZ': 2021, 'RI': 2022, 'VT': 2022,
    'MT': 2022, 'NJ': 2022, 'NM': 2022, 'NY': 2022, 'CT': 2023, 'MD': 2023,
    'MO': 2023, 'MN': 2024, 'OH': 2024,
}
NEVER_LEGAL = ['AL','AR','DC','DE','FL','GA','HI','IA','ID','IN','KS','KY','LA','ME',
               'MN','MS','NC','ND','NE','NH','OH','OK','PA','SC','SD','TN','TX','UT',
               'VA','WI','WV','WY']

# ── State polygons (Albers CONUS, meters) ──
states = gpd.read_file(os.path.join(DATA, 'us_states_shp', 'cb_2023_us_state_500k.shp'))
CONUS_EXCLUDE = {'AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP'}
states_conus = states[~states['STUSPS'].isin(CONUS_EXCLUDE)].to_crs(epsg=5070)
states_conus = states_conus.set_index('STUSPS')

# ── County centroids, ALL US counties (Gazetteer already covers everything) ──
gaz = pd.read_csv(os.path.join(DATA, 'gaz', '2023_Gaz_counties_national.txt'), sep='\t')
gaz.columns = [c.strip() for c in gaz.columns]
gaz['GEOID'] = gaz['GEOID'].astype(str).str.zfill(5)
gaz['USPS'] = gaz['USPS'].str.strip()
gaz_conus = gaz[~gaz['USPS'].isin(CONUS_EXCLUDE)].copy()
gdf_county = gpd.GeoDataFrame(gaz_conus, geometry=gpd.points_from_xy(gaz_conus['INTPTLONG'], gaz_conus['INTPTLAT']),
                               crs='EPSG:4326').to_crs(epsg=5070)

# ── Overdose raw data (already pulled nationwide, both years) ──
with open(os.path.join(DATA, 'overdose_county_raw.json'), encoding='utf-8-sig') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()

def relevant_border_geom(state_usps, is_never_legal, legal_states_asof):
    """Union of the state's shared boundary with all ADJACENT states on the
    other side of the legal/never-legal divide, as of a given year."""
    geom = states_conus.loc[state_usps, 'geometry']
    neighbors = states_conus[states_conus.geometry.touches(geom)]
    if is_never_legal:
        relevant = neighbors[neighbors.index.isin(legal_states_asof)]
    else:
        relevant = neighbors[neighbors.index.isin(NEVER_LEGAL)]
    if len(relevant) == 0:
        return None, []
    shared = [geom.boundary.intersection(g.boundary) for g in relevant.geometry]
    shared = [s for s in shared if not s.is_empty]
    if not shared:
        return None, []
    return unary_union(shared), list(relevant.index)

results = []
for year in [2020, 2023]:
    cutoff = 2019 if year == 2020 else year
    legal_states_asof = [s for s, y in LEGAL_STATES.items() if y <= cutoff]

    for state in NEVER_LEGAL:
        if state not in states_conus.index:
            continue
        border, neighbors = relevant_border_geom(state, True, legal_states_asof)
        if border is None:
            continue
        counties = gdf_county[gdf_county['USPS'] == state]
        for _, c in counties.iterrows():
            d_m = c.geometry.distance(border)
            results.append(dict(fips5=c['GEOID'], state=state, side='never_legal', year=year,
                                 dist_mi=d_m / 1609.344, nearest_neighbors=','.join(neighbors)))

    for state in legal_states_asof:
        if state not in states_conus.index:
            continue
        border, neighbors = relevant_border_geom(state, False, [])
        if border is None:
            continue
        counties = gdf_county[gdf_county['USPS'] == state]
        for _, c in counties.iterrows():
            d_m = c.geometry.distance(border)
            results.append(dict(fips5=c['GEOID'], state=state, side='legal', year=year,
                                 dist_mi=d_m / 1609.344, nearest_neighbors=','.join(neighbors)))
    print(f"year {year}: done ({len(results)} cumulative rows)")

dist_df = pd.DataFrame(results)
dist_df.to_csv(os.path.join(DATA, 'rdd_border_distances.csv'), index=False)
print(f"\nSaved rdd_border_distances.csv ({len(dist_df)} rows)")

# ── Merge overdose availability, restrict to counties with non-suppressed
#    data in BOTH years (needed for diff-in-disc), pivot wide by side ──
wide = dist_df.pivot_table(index=['fips5', 'state', 'side', 'nearest_neighbors'],
                            columns='year', values='dist_mi').reset_index()
wide.columns = [str(c) for c in wide.columns]
wide = wide.merge(od, on='fips5', how='left')
wide['has_both_years'] = wide['od_2020'].notna() & wide['od_2023'].notna()

print("\n" + "="*70)
print("FEASIBILITY: counties within bandwidth, by side, with non-suppressed")
print("overdose data in BOTH 2020 and 2023 (needed for diff-in-disc)")
print("="*70)
for bw in [10, 25, 50, 75, 100]:
    for yrcol in ['2020', '2023']:
        sub = wide[wide[yrcol] <= bw]
        n_nl = len(sub[sub['side'] == 'never_legal'])
        n_nl_ok = len(sub[(sub['side'] == 'never_legal') & sub['has_both_years']])
        n_lg = len(sub[sub['side'] == 'legal'])
        n_lg_ok = len(sub[(sub['side'] == 'legal') & sub['has_both_years']])
        print(f"bw={bw:3d}mi  as-of-{yrcol}  never-legal side: {n_nl_ok:4d}/{n_nl:4d} usable  |  "
              f"legal side: {n_lg_ok:4d}/{n_lg:4d} usable")
    print()
