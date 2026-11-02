# ============================================================
# Re-check of the CO+WA dispensary-revenue border analysis using the
# user-specified 70-mile threshold (originally analyze_revenue_border.py
# used 50mi, matching the rest of this project's convention). Treatment
# = county within 70mi of a non-legal-state border (captive cross-border
# demand); control = interior counties of the same legal states (CO,
# WA -- the only 2 states with a confirmed, usable county-level revenue
# data source; see project memory for why MA/NY/IL/OH/ME were ruled
# out). Same panel/data as analyze_revenue_border.py, just re-thresholded.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'border revenue', 'data')
TABLES = os.path.join(BASE, 'tables')
THRESH = 70

df = pd.read_csv(os.path.join(DATA, 'county_revenue_panel.csv'))
df = df[df['revenue_per_capita'] > 0].copy()
df['log_population'] = np.log(df['population'])
df['border_70mi'] = (df['dist_nonlegal_mi'] <= THRESH).astype(int)
df['county_g'] = pd.Categorical(df['fips5']).codes

print(f"Panel: {len(df)} county-year obs, {df['fips5'].nunique()} counties, states {df['state_abbr'].value_counts().to_dict()}")
print(f"Border (<= {THRESH}mi): {df.loc[df['border_70mi']==1,'fips5'].nunique()} counties, "
      f"interior: {df.loc[df['border_70mi']==0,'fips5'].nunique()} counties")
print(df.loc[df['border_70mi']==1, ['county_key','state_abbr','dist_nonlegal_mi']].drop_duplicates().to_string(index=False))

print("\nUNCONDITIONAL: revenue per capita, 70mi border vs interior")
print(df.groupby('border_70mi')['revenue_per_capita'].agg(['mean', 'median', 'std', 'count']))
print("\nBy state:")
print(df.groupby(['state_abbr', 'border_70mi'])['revenue_per_capita'].mean().unstack().round(2))

print("\n" + "=" * 70)
print("REGRESSIONS -- outcome: log(revenue per capita), 70-mile threshold")
print("=" * 70)
specs = {
    'No controls':                     'log_revenue_per_capita ~ border_70mi',
    '+ log(pop) + year FE + state FE': 'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year) + C(state_abbr)',
}
rows = []
for name, formula in specs.items():
    m = smf.ols(formula, data=df).fit(cov_type='HC1')
    mc = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:40s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(spec=name, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

# outlier check -- does Sedgwick (the single CO county driving the 50mi result) still dominate at 70mi?
print("\n" + "=" * 70)
print("Which CO counties fall within 70mi? (outlier-composition check)")
print("=" * 70)
co_border = df[(df['state_abbr']=='CO') & (df['border_70mi']==1)][['county_key','dist_nonlegal_mi']].drop_duplicates()
print(co_border.to_string(index=False))

sub = df[df['county_key'] != 'SEDGWICK']
sub['county_g'] = pd.Categorical(sub['fips5']).codes
formula = 'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year) + C(state_abbr)'
m = smf.ols(formula, data=sub).fit(cov_type='HC1')
mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['county_g']})
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
print(f"\n[excl. Sedgwick] + full controls          b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
rows.append(dict(spec='[excl. Sedgwick outlier] + log(pop) + year FE + state FE', coef=b, se_hc1=se, p_hc1=p,
                  se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'revenue_border_70mi_results.csv'), index=False)
print(f"\nSaved to {TABLES}")
