# ============================================================
# CORRECTED crop-substitution design, replacing analyze_crop_substitution_or.py's
# "has cultivation license" treatment (which had no border/interior
# geography at all) with the entry-game/border-competition treatment:
# TREATMENT = Oregon counties within 70 miles of Idaho (Oregon's only
# non-legal-state neighbor -- OR borders WA/CA/NV, all legal, and ID,
# not legal); CONTROL = interior Oregon counties, >70mi from Idaho.
# Outcome is the same harvested-cropland measure as before.
#
# Note discovered while building this: Jackson and Josephine counties
# (the two counties with by far the most cannabis cultivation canopy
# in the earlier, license-based analysis) are 313 and 357 miles from
# Idaho -- they sit near the CALIFORNIA border instead (itself legal),
# not near any captive non-legal-market demand. This means the earlier
# "has cultivation" proxy and this border-distance treatment are
# capturing genuinely different geography, not two measures of the
# same thing -- exactly why the corrected design matters.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
THRESH = 70

geo = pd.read_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'))
geo = geo[geo['state_abbr'] == 'OR'].copy()
geo['county_key'] = geo['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()
geo['border_70mi'] = (geo['dist_nonlegal_mi'] <= THRESH).astype(int)

nass_dir = os.path.join(BASE, '..', 'marijuana_new', 'health outcomes exploration', 'data')
crop = pd.read_csv(os.path.join(nass_dir, 'nass_cropland_harvested_or_county.csv'))
crop = crop[crop['VALUE'] != '(D)'].copy()
crop['VALUE'] = crop['VALUE'].str.replace(',', '').astype(float)
crop['YEAR'] = crop['YEAR'].astype(int)
crop['county_key'] = crop['COUNTY_NAME'].str.strip().str.upper()
crop_wide = crop.pivot_table(index='county_key', columns='YEAR', values='VALUE', aggfunc='first').reset_index()
crop_wide.columns = ['county_key'] + [f'cropland_{c}' for c in crop_wide.columns[1:]]

df = geo.merge(crop_wide, on='county_key', how='left')
df = df.dropna(subset=['cropland_2012', 'cropland_2022'])
df['log_population'] = np.log(df['population'])
df['log_change_cropland'] = np.log(df['cropland_2022'] / df['cropland_2012'])
df['pct_change_cropland'] = (df['cropland_2022'] - df['cropland_2012']) / df['cropland_2012']

print(f"Oregon counties with cropland data: {len(df)} of 36")
print(f"Border (<= {THRESH}mi of Idaho): {int(df['border_70mi'].sum())} counties -- "
      f"{df.loc[df['border_70mi']==1, 'county_key'].tolist()}")
print(f"Interior: {int((1-df['border_70mi']).sum())} counties")

print("\n" + "=" * 70)
print("Jackson/Josephine (the two highest-cultivation counties from the")
print("earlier license-based analysis) distance to nearest non-legal state:")
print("=" * 70)
print(df[df['county_key'].isin(['JACKSON', 'JOSEPHINE'])][['county_key', 'dist_nonlegal_mi', 'nearest_nonlegal_state']].to_string(index=False))

print("\n" + "=" * 70)
print("UNCONDITIONAL: cropland change 2012-2022, border vs interior")
print("=" * 70)
print(df.groupby('border_70mi')['pct_change_cropland'].agg(['mean', 'median', 'std', 'count']))
print("\nIndividual border counties:")
print(df.loc[df['border_70mi']==1, ['county_key', 'dist_nonlegal_mi', 'pct_change_cropland']].to_string(index=False))

print("\n" + "=" * 70)
print("REGRESSIONS -- outcome: log change in harvested cropland, 2012-2022")
print("(cross-sectional, n=36 max -- HC1 robust SEs, county clustering not")
print(" applicable: one observation per county, nothing to cluster)")
print("=" * 70)
rows = []
for name, formula in {
    'border_70mi, no controls': 'log_change_cropland ~ border_70mi',
    'border_70mi + log(pop) + log(land area)': 'log_change_cropland ~ border_70mi + log_population + np.log(land_area_sqmi)',
    'log(dist to non-legal border), no controls': 'log_change_cropland ~ np.log(dist_nonlegal_mi)',
    'log(dist) + log(pop) + log(land area)': 'log_change_cropland ~ np.log(dist_nonlegal_mi) + log_population + np.log(land_area_sqmi)',
}.items():
    m = smf.ols(formula, data=df).fit(cov_type='HC1')
    var = 'border_70mi' if 'border_70mi' in formula.split('~')[1] else 'np.log(dist_nonlegal_mi)'
    b, se, p = m.params[var], m.bse[var], m.pvalues[var]
    print(f"{name:48s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} R2={m.rsquared:.3f} n={int(m.nobs)}")
    rows.append(dict(spec=name, coef=b, se=se, p=p, r2=m.rsquared, n=int(m.nobs)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'crop_substitution_border_results.csv'), index=False)
df.to_csv(os.path.join(TABLES, 'crop_substitution_border_panel.csv'), index=False)
print(f"\nSaved to {TABLES}")
