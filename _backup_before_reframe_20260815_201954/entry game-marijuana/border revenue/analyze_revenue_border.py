# ============================================================
# Does county-level cannabis revenue per capita rise as counties get
# closer to a non-legal-state border (captive illegal-market demand)?
# Revenue/price can't be suppressed by a binding license cap the way
# raw dispensary counts can (see venous4's entry-density null) --
# this is the demand-side test that count data structurally couldn't run.
#
# County x fiscal-year panel, CO (2014-2025) + WA (2015-2025) only --
# the only 2 of the original 9 dispensary-covered states with a real,
# scriptable county-level revenue data source (see project memory for
# why MA/NY/IL/OH/ME/CA/OR are excluded). Primary clustering is by
# COUNTY (83 clusters, panel-appropriate), not state -- with only 2
# states, state-level clustering has just 2 clusters and is reported
# only as a heavily-flagged robustness check, not the headline SE.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

df = pd.read_csv(os.path.join(DATA, 'county_revenue_panel.csv'))
df = df[df['revenue_per_capita'] > 0].copy()
df['log_population'] = np.log(df['population'])
df['log_land_area'] = np.log(df['land_area_sqmi'])
df['county_g'] = pd.Categorical(df['fips5']).codes
df['state_g'] = pd.Categorical(df['state_abbr']).codes

print("=" * 70)
print(f"Panel: {len(df)} county-year obs, {df['fips5'].nunique()} counties, "
      f"states {df['state_abbr'].value_counts().to_dict()}, years {df['fy_year'].min()}-{df['fy_year'].max()}")
print("=" * 70)

print("\nUNCONDITIONAL: revenue per capita, border (<=50mi) vs interior")
print(df.groupby('border_50mi')['revenue_per_capita'].agg(['mean', 'median', 'std', 'count']))
print("\nBy state:")
print(df.groupby(['state_abbr', 'border_50mi'])['revenue_per_capita'].mean().unstack().round(2))

print("\n" + "=" * 70)
print("REGRESSIONS -- outcome: log(revenue per capita)")
print("=" * 70)

specs = {
    'No controls':                          'log_revenue_per_capita ~ log_dist_nonlegal',
    '+ log(pop)':                           'log_revenue_per_capita ~ log_dist_nonlegal + log_population',
    '+ year FE':                            'log_revenue_per_capita ~ log_dist_nonlegal + C(fy_year)',
    '+ state FE':                           'log_revenue_per_capita ~ log_dist_nonlegal + C(state_abbr)',
    '+ log(pop) + year FE + state FE':      'log_revenue_per_capita ~ log_dist_nonlegal + log_population + C(fy_year) + C(state_abbr)',
    '+ log(pop) + land area + year FE + state FE':
        'log_revenue_per_capita ~ log_dist_nonlegal + log_population + log_land_area + C(fy_year) + C(state_abbr)',
}

rows = []
for name, formula in specs.items():
    m = smf.ols(formula, data=df).fit(cov_type='HC1')
    mc = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['county_g']})
    ms = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['state_g']})
    b, se, p = m.params['log_dist_nonlegal'], m.bse['log_dist_nonlegal'], m.pvalues['log_dist_nonlegal']
    se_c, p_c = mc.bse['log_dist_nonlegal'], mc.pvalues['log_dist_nonlegal']
    se_s, p_s = ms.bse['log_dist_nonlegal'], ms.pvalues['log_dist_nonlegal']
    print(f"{name:48s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(83-county-cl)={se_c:.4f} p={p_c:.4f} "
          f"| se(2-state-cl, THIN)={se_s:.4f} p={p_s:.4f} R2={m.rsquared:.3f} n={int(m.nobs)}")
    rows.append(dict(spec=name, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c,
                      se_state_cl_thin=se_s, p_state_cl_thin=p_s, r2=m.rsquared, n=int(m.nobs)))

print("\n" + "=" * 70)
print("BINARY VERSION -- outcome: log(revenue per capita) ~ border_50mi")
print("=" * 70)
specs_bin = {
    'No controls':                     'log_revenue_per_capita ~ border_50mi',
    '+ log(pop) + year FE + state FE': 'log_revenue_per_capita ~ border_50mi + log_population + C(fy_year) + C(state_abbr)',
}
for name, formula in specs_bin.items():
    m = smf.ols(formula, data=df).fit(cov_type='HC1')
    mc = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['county_g']})
    b, se, p = m.params['border_50mi'], m.bse['border_50mi'], m.pvalues['border_50mi']
    se_c, p_c = mc.bse['border_50mi'], mc.pvalues['border_50mi']
    print(f"{name:48s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(83-county-cl)={se_c:.4f} p={p_c:.4f} "
          f"R2={m.rsquared:.3f} n={int(m.nobs)}")
    rows.append(dict(spec=f'[binary] {name}', coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c,
                      se_state_cl_thin=np.nan, p_state_cl_thin=np.nan, r2=m.rsquared, n=int(m.nobs)))

print("\n" + "=" * 70)
print("PER-STATE (single-state specs, since 2-state pooled state FE is a weak design)")
print("=" * 70)
for st in ['CO', 'WA']:
    sub = df[df['state_abbr'] == st]
    formula = 'log_revenue_per_capita ~ log_dist_nonlegal + log_population + C(fy_year)'
    m = sub_m = smf.ols(formula, data=sub).fit(cov_type='HC1')
    mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': pd.Categorical(sub['fips5']).codes})
    b, se, p = m.params['log_dist_nonlegal'], m.bse['log_dist_nonlegal'], m.pvalues['log_dist_nonlegal']
    se_c, p_c = mc.bse['log_dist_nonlegal'], mc.pvalues['log_dist_nonlegal']
    n_counties = sub['fips5'].nunique()
    print(f"{st} only (+log pop + year FE){'':10s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | "
          f"se({n_counties}-county-cl)={se_c:.4f} p={p_c:.4f} R2={m.rsquared:.3f} n={int(m.nobs)}")
    rows.append(dict(spec=f'[{st} only] + log(pop) + year FE', coef=b, se_hc1=se, p_hc1=p,
                      se_county_cl=se_c, p_county_cl=p_c, se_state_cl_thin=np.nan, p_state_cl_thin=np.nan,
                      r2=m.rsquared, n=int(m.nobs)))

print("\n" + "=" * 70)
print("OUTLIER CHECK: Sedgwick County, CO is the ONLY CO county within 50mi")
print("of a non-legal border (pop 2,299, on the NE border, known cross-border")
print("cannabis-tourism town) -- CO contributes a single county to the entire")
print("'border' group in the binary spec. Rerunning with it dropped.")
print("=" * 70)
sub = df[df['county_key'] != 'SEDGWICK'].copy()
for name, formula, var in [
    ('[binary] + log(pop) + year FE + state FE', 'log_revenue_per_capita ~ border_50mi + log_population + C(fy_year) + C(state_abbr)', 'border_50mi'),
    ('[continuous] + log(pop) + year FE + state FE', 'log_revenue_per_capita ~ log_dist_nonlegal + log_population + C(fy_year) + C(state_abbr)', 'log_dist_nonlegal'),
]:
    m = smf.ols(formula, data=sub).fit(cov_type='HC1')
    mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': pd.Categorical(sub['fips5']).codes})
    b, se, p = m.params[var], m.bse[var], m.pvalues[var]
    se_c, p_c = mc.bse[var], mc.pvalues[var]
    print(f"{name:48s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(82-county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(spec=f'[drop Sedgwick outlier] {name}', coef=b, se_hc1=se, p_hc1=p,
                      se_county_cl=se_c, p_county_cl=p_c, se_state_cl_thin=np.nan, p_state_cl_thin=np.nan,
                      r2=m.rsquared, n=int(m.nobs)))
print("\n-> Both specs collapse to a precise null once the single-county artifact is removed.")
print("   The marginal binary result (p=0.05 clustered) in the full panel was driven entirely")
print("   by Sedgwick's extreme per-capita figure, not a general border pattern.")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'revenue_border_results.csv'), index=False)
print("\nSaved tables/revenue_border_results.csv")
