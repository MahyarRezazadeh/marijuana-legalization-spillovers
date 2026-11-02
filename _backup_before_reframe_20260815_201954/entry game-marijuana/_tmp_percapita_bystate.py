import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('border revenue/data/county_revenue_panel.csv')
df['log_revenue_per_capita'] = np.log(df['revenue_per_capita'])
df['log_population'] = np.log(df['population'])
df['border_70mi'] = (df['dist_nonlegal_mi'] <= 70).astype(int)
df = df[df['revenue_per_capita'] > 0]

for st in ['CO', 'WA']:
    sub = df[df.state_abbr == st].copy()
    sub['g'] = pd.Categorical(sub['fips5']).codes
    formula = 'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)'
    m = smf.ols(formula, data=sub).fit(cov_type='HC1')
    mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
    b = m.params['border_70mi']
    p_hc1 = m.pvalues['border_70mi']
    p_cl = mc.pvalues['border_70mi']
    n = int(m.nobs)
    print(st, 'only revenue/capita: b=', round(b, 4), 'p(HC1)=', round(p_hc1, 4), 'p(cl)=', round(p_cl, 4), 'n=', n)
