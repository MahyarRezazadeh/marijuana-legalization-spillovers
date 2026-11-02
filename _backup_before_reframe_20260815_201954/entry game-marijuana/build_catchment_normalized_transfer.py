# ============================================================
# Normalizes the fiscal-transfer comparison (build_regime_transfer_
# comparison.py) by CATCHMENT POPULATION -- the true UNION of
# non-legal-state counties within 70mi of any of a regime's border
# counties, not a per-county sum (which would double-count non-legal
# counties that sit near more than one border county, an overlap risk
# especially for CO+WA's 12 border counties across UT/WY/NE).
#
# This tests whether CO+WA's larger raw dollar transfer is just a
# market-size (catchment population) artifact, or whether the capped
# regime is also more efficient per potential cross-border customer.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')
DATA = os.path.join(BASE, 'data')
EARTH_RADIUS_MI = 3958.8
REF_DATE = pd.Timestamp('2025-12-31')

# ---------- National county universe: centroids + population + legal-state map ----------
centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)
pop = pd.read_csv(os.path.join(DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop = pop[pop['SUMLEV'] == 50].copy()
pop['fips5'] = pop['STATE'].astype(str).str.zfill(2) + pop['COUNTY'].astype(str).str.zfill(3)
pop = pop[['fips5', 'POPESTIMATE2023']].rename(columns={'POPESTIMATE2023': 'population'})
centroids = centroids.merge(pop, on='fips5', how='left')

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
legal_states_asof = set(legal.loc[legal['retail_date'].notna() & (legal['retail_date'] <= REF_DATE), 'state_abbr'])
nonlegal = centroids[~centroids['state_abbr'].isin(legal_states_asof)].dropna(subset=['population']).copy()
print(f"Non-legal county universe: {len(nonlegal)} counties, total pop {nonlegal['population'].sum():,.0f}")

nonlegal_rad = np.radians(nonlegal[['lat', 'lon']].to_numpy())
tree = BallTree(nonlegal_rad, metric='haversine')
radius_rad = 70 / EARTH_RADIUS_MI

def catchment_population(border_fips_list):
    """Union of non-legal counties within 70mi of ANY border county in the list -- no double counting."""
    border_pts = centroids[centroids['fips5'].isin(border_fips_list)]
    own_rad = np.radians(border_pts[['lat', 'lon']].to_numpy())
    indices = tree.query_radius(own_rad, r=radius_rad)
    union_idx = set()
    for idx_list in indices:
        union_idx.update(idx_list)
    union_idx = sorted(union_idx)
    catchment = nonlegal.iloc[union_idx]
    return catchment['population'].sum(), catchment[['fips5', 'state_abbr', 'population']]

# ---------- Border county FIPS lists (from the existing corrected panels) ----------
perstore = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel_CORRECTED.csv'), dtype={'fips5': str})
cowa_border_fips = perstore.loc[(perstore['fy_year'] == 2025) & (perstore['border_70mi'] == 1), 'fips5'].unique().tolist()

mt = pd.read_csv(os.path.join(TABLES, 'montana_revenue_perstore_cross.csv'), dtype={'fips5': str})
mt_border_fips = mt.loc[mt['border_70mi'] == 1, 'fips5'].unique().tolist()

print(f"\nCO+WA border counties: {len(cowa_border_fips)}")
print(f"MT border counties: {len(mt_border_fips)}")

cowa_catchment_pop, cowa_catchment_detail = catchment_population(cowa_border_fips)
mt_catchment_pop, mt_catchment_detail = catchment_population(mt_border_fips)

print(f"\nCO+WA catchment (union, non-legal counties within 70mi of any CO/WA border county):")
print(f"  {len(cowa_catchment_detail)} non-legal counties, states: {sorted(cowa_catchment_detail['state_abbr'].unique())}")
print(f"  Total catchment population: {cowa_catchment_pop:,.0f}")

print(f"\nMontana catchment (union, non-legal counties within 70mi of any MT border county):")
print(f"  {len(mt_catchment_detail)} non-legal counties, states: {sorted(mt_catchment_detail['state_abbr'].unique())}")
print(f"  Total catchment population: {mt_catchment_pop:,.0f}")

# ---------- Combine with the already-estimated dollar transfers ----------
transfer = pd.read_csv(os.path.join(TABLES, 'regime_transfer_comparison.csv')).set_index('regime')

cowa_attrib = transfer.loc['Capped (CO+WA pooled)', 'attributable']
mt_attrib = transfer.loc['Moratorium (Montana)', 'attributable']

cowa_per_capita = cowa_attrib / cowa_catchment_pop
mt_per_capita = mt_attrib / mt_catchment_pop

print("\n" + "=" * 70)
print("CATCHMENT-NORMALIZED FISCAL TRANSFER")
print("=" * 70)
print(f"\nCapped (CO+WA): ${cowa_attrib:,.0f} transferred / {cowa_catchment_pop:,.0f} catchment pop "
      f"= ${cowa_per_capita:.2f} per catchment resident")
print(f"Moratorium (MT): ${mt_attrib:,.0f} transferred / {mt_catchment_pop:,.0f} catchment pop "
      f"= ${mt_per_capita:.2f} per catchment resident")
print(f"\nRatio, capped:moratorium, raw dollars: {cowa_attrib/mt_attrib:.2f}x")
print(f"Ratio, capped:moratorium, catchment population: {cowa_catchment_pop/mt_catchment_pop:.2f}x")
print(f"Ratio, capped:moratorium, $ per catchment resident: {cowa_per_capita/mt_per_capita:.2f}x")

out = pd.DataFrame([
    dict(regime='Capped (CO+WA pooled)', attributable=cowa_attrib, catchment_population=cowa_catchment_pop,
         transfer_per_catchment_resident=cowa_per_capita, n_nonlegal_counties=len(cowa_catchment_detail)),
    dict(regime='Moratorium (Montana)', attributable=mt_attrib, catchment_population=mt_catchment_pop,
         transfer_per_catchment_resident=mt_per_capita, n_nonlegal_counties=len(mt_catchment_detail)),
])
out.to_csv(os.path.join(TABLES, 'catchment_normalized_transfer.csv'), index=False)
print(f"\nSaved {os.path.join(TABLES, 'catchment_normalized_transfer.csv')}")
