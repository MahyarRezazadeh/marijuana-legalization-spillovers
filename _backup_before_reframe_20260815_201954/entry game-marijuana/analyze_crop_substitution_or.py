# ============================================================
# Crop substitution, Oregon: do counties with more cannabis cultivation
# canopy show a bigger decline in traditional harvested cropland?
#
# IMPORTANT DESIGN LIMITATION, disclosed upfront: unlike the land-value
# analysis, this is NOT an event study and has NO parallel-trends check.
# OLCC's license dataset only records each license's CURRENT renewal
# window (effective_date is overwhelmingly 2023-2026 -- it is a live
# registry snapshot, not a historical record), so there is no way to
# recover which counties had cultivation licenses active in 2012 or
# 2017. The cannabis "treatment" here is necessarily a CURRENT
# cross-sectional intensity measure, correlated against the CHANGE in
# harvested cropland from 2012 (entirely pre-legalization) to 2022
# (mature market). This is a correlational cross-sectional design, not
# a causal DiD -- read the results accordingly. It cannot rule out
# reverse causation (cultivation may have located in counties where
# traditional cropland was already declining for unrelated reasons --
# exactly the pattern Schwab & Butsic describe in Humboldt, where
# cannabis moved into marginal, not-prime agricultural land).
# ============================================================
import os
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

# canopy sq ft by (tier, canopy_type) -- upper bound of each tier's regulatory cap (OAR 845-025-2040).
# Mixed producers use a 4:1 outdoor:indoor equivalency; approximated here as indoor_cap + outdoor_cap
# for the same tier (upper-bound proxy, not exact -- acceptable for a relative-intensity measure only).
CANOPY_SQFT = {
    ('Tier I', 'Indoor'): 5_000, ('Tier I', 'Outdoor'): 20_000, ('Tier I', 'Mixed'): 25_000,
    ('Tier II', 'Indoor'): 10_000, ('Tier II', 'Outdoor'): 40_000, ('Tier II', 'Mixed'): 50_000,
    ('Micro Tier I', 'Indoor'): 625, ('Micro Tier I', 'Outdoor'): 2_500, ('Micro Tier I', 'Mixed'): 3_125,
    ('Micro Tier II', 'Indoor'): 1_250, ('Micro Tier II', 'Outdoor'): 5_000, ('Micro Tier II', 'Mixed'): 6_250,
}

with open(os.path.join(DATA, 'raw', 'or_cannabis_licenses.json'), encoding='utf-8') as f:
    lic = pd.DataFrame(json.load(f))
prod = lic[lic['license_type'] == 'RECREATIONAL PRODUCER'].copy()
prod = prod[prod['license_expired'] == 'No']  # currently active only
prod['canopy_sqft'] = prod.apply(lambda r: CANOPY_SQFT.get((r['tier'], r['canopy_type']), np.nan), axis=1)
print(f"Active producer licenses: {len(prod)}, unmatched tier/canopy_type combos: {prod['canopy_sqft'].isna().sum()}")

canopy_by_county = prod.groupby('county', as_index=False).agg(
    canopy_sqft=('canopy_sqft', 'sum'), n_licenses=('license_number', 'count'))
canopy_by_county['county_key'] = canopy_by_county['county'].str.strip().str.upper()

crop = pd.read_csv(os.path.join(DATA, 'nass_cropland_harvested_or_county.csv'))
crop = crop[crop['VALUE'] != '(D)'].copy()
crop['VALUE'] = crop['VALUE'].str.replace(',', '').astype(float)
crop['YEAR'] = crop['YEAR'].astype(int)
crop['county_key'] = crop['COUNTY_NAME'].str.strip().str.upper()
crop_wide = crop.pivot_table(index='county_key', columns='YEAR', values='VALUE', aggfunc='first').reset_index()
crop_wide.columns = ['county_key'] + [f'cropland_{c}' if c != 'county_key' else c for c in crop_wide.columns[1:]]

geo = pd.read_csv(os.path.join(BASE, '..', '..', 'entry game-marijuana', 'data', 'county_universe_with_distance.csv'))
geo = geo[geo['state_abbr'] == 'OR'].copy()
geo['county_key'] = geo['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()

df = geo.merge(crop_wide, on='county_key', how='left').merge(canopy_by_county, on='county_key', how='left')
df['canopy_sqft'] = df['canopy_sqft'].fillna(0)
df['n_licenses'] = df['n_licenses'].fillna(0)
df['has_cultivation'] = (df['n_licenses'] > 0).astype(int)
df['log_canopy'] = np.log(df['canopy_sqft'] + 1)
df['log_population'] = np.log(df['population'])

df = df.dropna(subset=['cropland_2012', 'cropland_2022'])
print(f"\nOregon counties with cropland data both years: {len(df)} of 36")
print(f"Counties with any active cultivation license: {int(df['has_cultivation'].sum())}")

df['pct_change_cropland'] = (df['cropland_2022'] - df['cropland_2012']) / df['cropland_2012']
df['log_change_cropland'] = np.log(df['cropland_2022'] / df['cropland_2012'])

print("\n" + "=" * 70)
print("UNCONDITIONAL: cropland change 2012-2022, by cultivation presence")
print("=" * 70)
print(df.groupby('has_cultivation')['pct_change_cropland'].agg(['mean', 'median', 'std', 'count']))

print("\n" + "=" * 70)
print("CROSS-SECTIONAL REGRESSIONS (correlational -- see design note above)")
print("outcome: log change in harvested cropland acres, 2012-2022")
print("=" * 70)
specs = {
    'has_cultivation (binary), no controls': 'log_change_cropland ~ has_cultivation',
    'has_cultivation + log(pop) + log(land area)': 'log_change_cropland ~ has_cultivation + log_population + np.log(land_area_sqmi)',
    'log(canopy sqft), no controls': 'log_change_cropland ~ log_canopy',
    'log(canopy sqft) + log(pop) + log(land area)': 'log_change_cropland ~ log_canopy + log_population + np.log(land_area_sqmi)',
}
rows = []
for name, formula in specs.items():
    m = smf.ols(formula, data=df).fit(cov_type='HC1')
    var = 'has_cultivation' if 'has_cultivation' in formula.split('~')[1].split('+')[0] else 'log_canopy'
    b, se, p = m.params[var], m.bse[var], m.pvalues[var]
    print(f"{name:48s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} R2={m.rsquared:.3f} n={int(m.nobs)}")
    rows.append(dict(spec=name, coef=b, se=se, p=p, r2=m.rsquared, n=int(m.nobs)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'crop_substitution_or_results.csv'), index=False)
print("\nSaved tables/crop_substitution_or_results.csv")

print("\n" + "=" * 70)
print("Top 10 counties by current cultivation canopy (context)")
print("=" * 70)
print(df.sort_values('canopy_sqft', ascending=False)[['county_key', 'canopy_sqft', 'n_licenses', 'pct_change_cropland']].head(10).to_string(index=False))
