# ============================================================
# Placebo check: does the border revenue-per-store premium show up in
# GENERAL retail (all industries, Colorado Dept of Revenue), or is it
# specific to cannabis? If border towns just do more retail business
# generally (tourism, price arbitrage, any cross-border shopping),
# general retail sales per retailer should show the same border
# premium. If the effect is cannabis-specific, it should not.
# Colorado only -- no equivalent WA general-retail dataset could be
# reached in the time budgeted for this check (data.lcb.wa.gov is
# unreachable from this environment; data.wa.gov's Taxable Retail
# Sales is an ASPX report tool, not a queryable API).
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'placebo')
TABLES = os.path.join(BASE, 'tables')

df = pd.read_json(os.path.join(RAW, 'co_total_retail_by_county.json'))
df['year'] = df['year'].astype(float).astype(int)
df['month'] = df['month'].astype(int)
df['retailsales'] = pd.to_numeric(df['retailsales'], errors='coerce')
df['numberofretailers'] = pd.to_numeric(df['numberofretailers'], errors='coerce')
df = df[df['county'] != 'Total'].copy()

# complete-12-months calendar years only, same convention as the cannabis revenue build
counts = df.groupby(['county', 'year']).size().rename('n_months')
annual = df.groupby(['county', 'year']).agg(retailsales=('retailsales', 'sum'),
                                             retailers=('numberofretailers', 'mean')).reset_index()
annual = annual.merge(counts, on=['county', 'year'])
annual = annual[annual['n_months'] == 12].drop(columns='n_months')
annual['county_key'] = annual['county'].str.strip().str.upper()

geo = pd.read_csv(os.path.join(BASE, 'data', 'county_universe_all23_with_distance.csv'), dtype={'fips5': str})
geo = geo[geo['state_abbr'] == 'CO'].copy()
geo['county_key'] = geo['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()
geo = geo[['fips5', 'county_key', 'population', 'dist_nonlegal_mi']]

panel = annual.merge(geo, on='county_key', how='inner')
panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= 70).astype(int)
panel['log_sales_per_capita'] = np.log(panel['retailsales'] / panel['population'])
panel['log_population'] = np.log(panel['population'])
panel = panel.reset_index(drop=True)
panel['g'] = pd.Categorical(panel['fips5']).codes

print(f"Panel (per-capita, all years): {len(panel)} county-year obs, {panel['fips5'].nunique()} counties, "
      f"years {sorted(panel['year'].unique())}")
print(f"Border: {panel.loc[panel['border_70mi']==1,'fips5'].nunique()} counties, "
      f"Interior: {panel.loc[panel['border_70mi']==0,'fips5'].nunique()} counties")

# CDOR's numberofretailers field is only populated starting 2025 (NaN in all earlier years) --
# a real data limitation, not a bug: "sales per retailer" can only be a single-year (2025)
# cross-section, unlike "sales per capita" which is available for the full 2016-2025 panel.
panel_pr = panel[panel['retailers'].notna() & (panel['retailers'] > 0)].copy()
panel_pr['sales_per_retailer'] = panel_pr['retailsales'] / panel_pr['retailers']
panel_pr['log_sales_per_retailer'] = np.log(panel_pr['sales_per_retailer'])
panel_pr = panel_pr.reset_index(drop=True)
panel_pr['g'] = pd.Categorical(panel_pr['fips5']).codes
print(f"\nPanel (per-retailer, 2025 only -- numberofretailers unavailable before 2025): "
      f"{len(panel_pr)} obs, years {sorted(panel_pr['year'].unique())}")

print("\n" + "=" * 70)
print("PLACEBO -- outcome: log(GENERAL retail sales per retailer), Colorado, 2025 cross-section only")
print("=" * 70)
formula = 'log_sales_per_retailer ~ border_70mi + log_population'
m = smf.ols(formula, data=panel_pr).fit(cov_type='HC1')
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
print(f"{'+ log(pop), single cross-section':32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} n={int(m.nobs)} "
      f"(NOTE: n=11, single year -- no clustering possible, HC1 only, treat as illustrative)")

print("\n" + "=" * 70)
print("PLACEBO -- outcome: log(GENERAL retail sales per capita), Colorado")
print("=" * 70)
for name, formula, var in [
    ('No controls', 'log_sales_per_capita ~ border_70mi', 'border_70mi'),
    ('+ log(pop) + year FE', 'log_sales_per_capita ~ border_70mi + log_population + C(year)', 'border_70mi'),
]:
    m = smf.ols(formula, data=panel).fit(cov_type='HC1')
    mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['g']})
    b, se, p = m.params[var], m.bse[var], m.pvalues[var]
    se_c, p_c = mc.bse[var], mc.pvalues[var]
    print(f"{name:32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")

panel.to_csv(os.path.join(TABLES, 'placebo_general_retail_co_panel.csv'), index=False)
print(f"\nSaved to {TABLES}")
