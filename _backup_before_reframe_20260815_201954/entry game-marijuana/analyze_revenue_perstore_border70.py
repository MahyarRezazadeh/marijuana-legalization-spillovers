# ============================================================
# Revenue PER DISPENSARY STORE (not per capita) under the same
# border-vs-interior (70mi) treatment design. This isolates whether
# individual outlets in border counties do more business, separate
# from how many people live there or how many stores have opened --
# a more direct test of "does each store absorb more captive
# cross-border demand" than the per-capita revenue check.
#
# Dispensary counts are cumulative-open-by-year, built from the same
# geocoded dispensary listing (dispensaries_master_geocoded.csv, CO
# and WA subset) used throughout venous4, point-in-polygon assigned to
# counties, counted as open if open_date <= the fiscal/calendar
# year-end matching each state's own revenue-reporting convention
# (CO: calendar year, cutoff Dec 31; WA: fiscal year Jul-Jun, cutoff
# Jun 30).
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

disp = pd.read_csv(os.path.join(DATA, 'dispensaries_master_geocoded.csv'))
disp = disp[disp['state'].isin(['CO', 'WA'])].dropna(subset=['lat', 'lon']).copy()
disp['open_date'] = pd.to_datetime(disp['open_date'], errors='coerce')
print(f"CO+WA geocoded dispensaries: {len(disp)} ({disp['open_date'].isna().sum()} missing open_date, treated as always-open)")

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
shp = shp[['GEOID', 'geometry']].rename(columns={'GEOID': 'fips5'})
disp_gdf = gpd.GeoDataFrame(disp, geometry=[Point(xy) for xy in zip(disp['lon'], disp['lat'])], crs='EPSG:4326')
disp_gdf = disp_gdf.to_crs(shp.crs)
joined = gpd.sjoin(disp_gdf, shp, how='left', predicate='within')
unmatched = joined['fips5'].isna().sum()
if unmatched:
    missing = joined[joined['fips5'].isna()][['uid', 'geometry']]
    nearest = gpd.sjoin_nearest(missing.set_geometry('geometry'), shp, how='left')
    fix = dict(zip(nearest['uid'], nearest['fips5_right'] if 'fips5_right' in nearest.columns else nearest['fips5']))
    joined.loc[joined['fips5'].isna(), 'fips5'] = joined.loc[joined['fips5'].isna(), 'uid'].map(fix)
print(f"Assigned to counties: {joined['fips5'].notna().sum()} of {len(disp)}")

rev = pd.read_csv(os.path.join(BASE, 'border revenue', 'data', 'county_revenue_panel.csv'), dtype={'fips5': str})
rev['fips5'] = rev['fips5'].str.zfill(5)  # source file has CO's fips5 unpadded (e.g. "8001" not "08001")
rev = rev[rev['revenue_per_capita'] > 0].copy()

rows = []
for _, r in rev.iterrows():
    fy = int(r['fy_year'])
    cutoff = pd.Timestamp(f'{fy}-12-31') if r['state_abbr'] == 'CO' else pd.Timestamp(f'{fy}-06-30')
    rows.append(cutoff)
rev['cutoff'] = rows

counts = []
for cutoff, grp in rev.groupby('cutoff'):
    open_asof = joined[joined['open_date'].isna() | (joined['open_date'] <= cutoff)]
    c = open_asof.groupby('fips5').size().rename('dispensary_count').reset_index()
    c['cutoff'] = cutoff
    counts.append(c)
counts = pd.concat(counts, ignore_index=True)

panel = rev.merge(counts, on=['fips5', 'cutoff'], how='left')
panel['dispensary_count'] = panel['dispensary_count'].fillna(0)
panel = panel[panel['dispensary_count'] > 0].copy()  # revenue-per-store undefined with zero stores
panel['revenue_per_store'] = panel['total_sales'] / panel['dispensary_count']
panel['log_revenue_per_store'] = np.log(panel['revenue_per_store'])
panel['log_population'] = np.log(panel['population'])
panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= 70).astype(int)
panel['county_g'] = pd.Categorical(panel['fips5']).codes

print(f"\nPanel with >=1 store: {len(panel)} county-year obs, {panel['fips5'].nunique()} counties")
print(f"Border: {panel.loc[panel['border_70mi']==1,'fips5'].nunique()} counties, "
      f"Interior: {panel.loc[panel['border_70mi']==0,'fips5'].nunique()} counties")

print("\nUNCONDITIONAL: revenue per store, border vs interior")
print(panel.groupby('border_70mi')['revenue_per_store'].agg(['mean', 'median', 'std', 'count']))
print("\nBy state:")
print(panel.groupby(['state_abbr', 'border_70mi'])['revenue_per_store'].mean().unstack().round(0))

print("\n" + "=" * 70)
print("REGRESSIONS -- outcome: log(revenue per store)")
print("=" * 70)
rows_out = []
for name, formula in {
    'No controls': 'log_revenue_per_store ~ border_70mi',
    '+ log(pop) + year FE + state FE': 'log_revenue_per_store ~ border_70mi + log_population + C(fy_year) + C(state_abbr)',
}.items():
    m = smf.ols(formula, data=panel).fit(cov_type='HC1')
    mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:40s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows_out.append(dict(spec=name, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

sub = panel[panel['county_key'] != 'SEDGWICK'].copy()
sub['county_g'] = pd.Categorical(sub['fips5']).codes
formula = 'log_revenue_per_store ~ border_70mi + log_population + C(fy_year) + C(state_abbr)'
m = smf.ols(formula, data=sub).fit(cov_type='HC1')
mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['county_g']})
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
print(f"{'[excl. Sedgwick] + full controls':40s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
rows_out.append(dict(spec='[excl. Sedgwick] + log(pop) + year FE + state FE', coef=b, se_hc1=se, p_hc1=p,
                      se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

res = pd.DataFrame(rows_out)
res.to_csv(os.path.join(TABLES, 'revenue_perstore_border70_results.csv'), index=False)
panel.to_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel.csv'), index=False)
print(f"\nSaved to {TABLES}")
