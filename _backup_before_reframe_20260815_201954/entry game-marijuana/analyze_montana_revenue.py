# ============================================================
# Montana border-vs-interior regression, revenue per capita.
# Montana is the highest-value state for this design geographically
# (it borders FOUR non-legal states: ID, ND, SD, WY), giving 11 border
# counties -- more than CO, WA, and OR combined.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

panel = pd.read_csv(os.path.join(TABLES, 'montana_revenue_percapita_panel.csv'), dtype={'fips5': str})
panel['county_g'] = pd.Categorical(panel['fips5']).codes

print("=" * 70)
print("MONTANA STANDALONE -- outcome: log(revenue per capita)")
print("=" * 70)
print(f"n={len(panel)}, {panel['fips5'].nunique()} counties, "
      f"border={panel.loc[panel['border_70mi']==1,'fips5'].nunique()}, "
      f"interior={panel.loc[panel['border_70mi']==0,'fips5'].nunique()}")
print(f"Border counties: {sorted(panel.loc[panel['border_70mi']==1,'county_name'].unique())}")

print("\nUNCONDITIONAL means:")
print(panel.groupby('border_70mi')['revenue_per_capita'].agg(['mean', 'median', 'std', 'count']))

rows = []
for name, formula in {
    'No controls': 'log_revenue_per_capita ~ border_70mi',
    '+ log(pop) + year FE': 'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)',
}.items():
    m = smf.ols(formula, data=panel).fit(cov_type='HC1')
    mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:32s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(spec=name, sample='MT only', coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c,
                      p_county_cl=p_c, n=int(m.nobs)))

# find which single county (if any) drives the result
print("\nBy border county (mean log revenue per capita vs. interior mean):")
interior_mean = panel.loc[panel['border_70mi']==0, 'log_revenue_per_capita'].mean()
print(f"Interior mean: {interior_mean:.3f}")
print(panel[panel['border_70mi']==1].groupby('county_name')['log_revenue_per_capita'].mean().sort_values(ascending=False))

# leave-one-out on the largest border county
top_border = panel[panel['border_70mi']==1].groupby('county_key')['total_sales'].sum().idxmax()
sub = panel[panel['county_key'] != top_border].copy()
sub['county_g'] = pd.Categorical(sub['fips5']).codes
m = smf.ols('log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)', data=sub).fit(cov_type='HC1')
mc = smf.ols('log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
print(f"\n[excl. {top_border}, largest border county by revenue] + log(pop) + year FE  "
      f"b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
rows.append(dict(spec=f'[excl. {top_border}] + log(pop) + year FE', sample='MT only', coef=b, se_hc1=se,
                  p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'montana_revenue_results.csv'), index=False)
print(f"\nSaved to {TABLES}")
