# ============================================================
# Pool Oregon's 2025 revenue-per-store cross-section into the main
# CO+WA revenue-per-store panel. This is methodologically valid even
# though OR contributes only one year: year FE absorb any OR-specific
# 2025 level shock alongside CO/WA's, and state FE absorb any
# state-level revenue-per-store difference, so OR's within-state
# border/interior contrast is identified the same way CO's and WA's
# are, just from a single year instead of many.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

cowa = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel.csv'), dtype={'fips5': str})
keep = ['fips5', 'state_abbr', 'county_key', 'fy_year', 'total_sales', 'population', 'dist_nonlegal_mi',
        'border_70mi', 'dispensary_count', 'revenue_per_store', 'log_revenue_per_store', 'log_population']
cowa = cowa[keep]

orx = pd.read_csv(os.path.join(TABLES, 'oregon_revenue_perstore_cross2025.csv'))
orx = orx[keep]

pooled = pd.concat([cowa, orx], ignore_index=True)
pooled['county_g'] = pd.Categorical(pooled['fips5']).codes

print(f"Pooled CO+WA+OR revenue-per-store panel: {len(pooled)} obs, {pooled['fips5'].nunique()} counties")
print(pooled.groupby('state_abbr')['fips5'].nunique())
print(f"Border counties: {pooled[pooled['border_70mi']==1].groupby('state_abbr')['fips5'].nunique().to_dict()}")

rows = []
print("\n" + "=" * 70)
print("POOLED CO+WA+OR -- outcome: log(revenue per store)")
print("=" * 70)
for name, formula in {
    'No controls': 'log_revenue_per_store ~ border_70mi',
    '+ log(pop) + year FE + state FE': 'log_revenue_per_store ~ border_70mi + log_population + C(fy_year) + C(state_abbr)',
}.items():
    m = smf.ols(formula, data=pooled).fit(cov_type='HC1')
    mc = smf.ols(formula, data=pooled).fit(cov_type='cluster', cov_kwds={'groups': pooled['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(spec=name, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

sub = pooled[~pooled['county_key'].str.upper().isin(['SEDGWICK', 'MALHEUR'])].copy()
sub['county_g'] = pd.Categorical(sub['fips5']).codes
formula = 'log_revenue_per_store ~ border_70mi + log_population + C(fy_year) + C(state_abbr)'
m = smf.ols(formula, data=sub).fit(cov_type='HC1')
mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['county_g']})
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
print(f"{'[excl. Sedgwick+Malheur] full controls':32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | "
      f"se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
rows.append(dict(spec='[excl. Sedgwick+Malheur] + log(pop) + year FE + state FE', coef=b, se_hc1=se, p_hc1=p,
                  se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

# by state separately (OR is single-year so this is just OR's cross-sectional coefficient again, for comparability)
for st in ['CO', 'WA', 'OR']:
    d = pooled[pooled['state_abbr'] == st].copy()
    d['county_g'] = pd.Categorical(d['fips5']).codes
    f = 'log_revenue_per_store ~ border_70mi + log_population' + (' + C(fy_year)' if st != 'OR' else '')
    m = smf.ols(f, data=d).fit(cov_type='HC1')
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    print(f"  [{st} only] b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} n={int(m.nobs)}")

pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'pooled_cowaOR_revenue_perstore_results.csv'), index=False)
pooled.to_csv(os.path.join(TABLES, 'pooled_cowaOR_revenue_perstore_panel.csv'), index=False)
print(f"\nSaved to {TABLES}")
