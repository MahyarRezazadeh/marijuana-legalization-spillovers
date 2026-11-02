# ============================================================
# Oregon border-vs-interior regressions (standalone), plus pooled
# CO+WA+OR revenue-per-capita panel and event study. Revenue-per-store
# for OR is a single 2025 cross-section (see build_oregon_panel_final.py
# for why), so it is reported standalone only, not pooled into the
# CO+WA per-store panel event study (which needs a real multi-year
# panel to estimate market-age bins).
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

# ---------- Oregon standalone: revenue per capita ----------
or_panel = pd.read_csv(os.path.join(TABLES, 'oregon_revenue_percapita_panel.csv'), dtype={'fips5': str})
or_panel['county_g'] = pd.Categorical(or_panel['fips5']).codes

print("=" * 70)
print("OREGON STANDALONE -- outcome: log(revenue per capita)")
print("=" * 70)
print(f"n={len(or_panel)}, border counties={sorted(or_panel.loc[or_panel['border_70mi']==1,'county_name'].unique())}")
rows = []
for name, formula in {
    'No controls': 'log_revenue_per_capita ~ border_70mi',
    '+ log(pop) + year FE': 'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)',
}.items():
    m = smf.ols(formula, data=or_panel).fit(cov_type='HC1')
    mc = smf.ols(formula, data=or_panel).fit(cov_type='cluster', cov_kwds={'groups': or_panel['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(spec=name, outcome='revenue_per_capita', sample='OR only', coef=b, se_hc1=se, p_hc1=p,
                      se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

# excl. Malheur (the OR equivalent of Sedgwick -- Ontario, OR sits directly across
# the Snake River from Fruitland/Payette, Idaho, and dominates OR's border-county revenue)
sub = or_panel[or_panel['county_key'] != 'MALHEUR'].copy()
sub['county_g'] = pd.Categorical(sub['fips5']).codes
m = smf.ols('log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)', data=sub).fit(cov_type='HC1')
mc = smf.ols('log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
print(f"{'[excl. Malheur] + log(pop) + year FE':32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | "
      f"se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
rows.append(dict(spec='[excl. Malheur] + log(pop) + year FE', outcome='revenue_per_capita', sample='OR only',
                  coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

# ---------- Oregon standalone: revenue per store (2025 cross-section) ----------
or_cross = pd.read_csv(os.path.join(TABLES, 'oregon_revenue_perstore_cross2025.csv'))
print("\n" + "=" * 70)
print("OREGON STANDALONE -- outcome: log(revenue per store), 2025 cross-section only")
print("=" * 70)
m = smf.ols('log_revenue_per_store ~ border_70mi + log_population', data=or_cross).fit(cov_type='HC1')
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
print(f"{'+ log(pop), single cross-section':32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} n={int(m.nobs)} "
      f"(NOTE: n=30, single year -- no clustering possible, HC1 only)")
rows.append(dict(spec='+ log(pop), single cross-section (2025)', outcome='revenue_per_store', sample='OR only',
                  coef=b, se_hc1=se, p_hc1=p, se_county_cl=np.nan, p_county_cl=np.nan, n=int(m.nobs)))

sub_cross = or_cross[or_cross['county_key'] != 'MALHEUR']
m = smf.ols('log_revenue_per_store ~ border_70mi + log_population', data=sub_cross).fit(cov_type='HC1')
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
print(f"{'[excl. Malheur] + log(pop), cross-section':32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} n={int(m.nobs)}")
rows.append(dict(spec='[excl. Malheur] + log(pop), cross-section (2025)', outcome='revenue_per_store',
                  sample='OR only', coef=b, se_hc1=se, p_hc1=p, se_county_cl=np.nan, p_county_cl=np.nan,
                  n=int(m.nobs)))

# ---------- Pooled CO+WA+OR: revenue per capita, market-age event study ----------
print("\n" + "=" * 70)
print("POOLED CO+WA+OR -- revenue per capita, border x market-age event study")
print("=" * 70)
cowa = pd.read_csv(os.path.join(BASE, 'border revenue', 'data', 'county_revenue_panel.csv'), dtype={'fips5': str})
cowa['fips5'] = cowa['fips5'].str.zfill(5)
cowa = cowa[cowa['revenue_per_capita'] > 0].copy()
cowa['log_population'] = np.log(cowa['population'])
cowa['border_70mi'] = (cowa['dist_nonlegal_mi'] <= 70).astype(int)
cowa = cowa[['fips5', 'county_key', 'state_abbr', 'fy_year', 'total_sales', 'population', 'dist_nonlegal_mi',
             'border_70mi', 'revenue_per_capita', 'log_revenue_per_capita', 'log_population']]

or_p = or_panel[['fips5', 'county_key', 'state_abbr', 'fy_year', 'total_sales', 'population', 'dist_nonlegal_mi',
                  'border_70mi', 'revenue_per_capita', 'log_revenue_per_capita', 'log_population']]

pooled = pd.concat([cowa, or_p], ignore_index=True)
pooled['county_g'] = pd.Categorical(pooled['fips5']).codes
print(f"Pooled panel: {len(pooled)} obs, {pooled['fips5'].nunique()} counties, states {sorted(pooled['state_abbr'].unique())}")
print(f"Border counties by state:")
print(pooled[pooled['border_70mi']==1].groupby('state_abbr')['fips5'].nunique())

launch_offset = {'CO': 2014, 'WA': 2015, 'OR': 2016}  # OR's OLCC data itself starts 2016 (see caveat)
pooled['event_year'] = pooled.apply(lambda r: int(r['fy_year']) - launch_offset[r['state_abbr']], axis=1)

print("\n--- Pooled level regression (no event study) ---")
for name, formula in {
    'No controls': 'log_revenue_per_capita ~ border_70mi',
    '+ log(pop) + year FE + state FE': 'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year) + C(state_abbr)',
}.items():
    m = smf.ols(formula, data=pooled).fit(cov_type='HC1')
    mc = smf.ols(formula, data=pooled).fit(cov_type='cluster', cov_kwds={'groups': pooled['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(spec=name, outcome='revenue_per_capita', sample='CO+WA+OR pooled', coef=b, se_hc1=se,
                      p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

pooled.to_csv(os.path.join(TABLES, 'pooled_cowaOR_revenue_percapita_panel.csv'), index=False)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'oregon_and_pooled_results.csv'), index=False)
print(f"\nSaved results to {TABLES}")
