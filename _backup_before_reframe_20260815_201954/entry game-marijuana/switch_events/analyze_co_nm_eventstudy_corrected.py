# ============================================================
# Verification rerun of analyze_co_nm_eventstudy.py using the corrected panel
# (co_nm_switch_panel_CORRECTED.csv). Colorado's dispensary registry was never
# affected by the Washington data-quality issue, so this should reproduce the
# original CO-NM results exactly; this script exists to confirm that directly
# rather than assume it.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(BASE, 'co_nm_switch_panel_CORRECTED.csv'), dtype={'fips5': str})

d = panel[panel['fy_year'] != 2022].copy()
d['post'] = (d['fy_year'] >= 2023).astype(int)
d['g'] = pd.Categorical(d['fips5']).codes
formula = 'log_revenue_per_store ~ nm_border_70mi*post + other_border_70mi + log_population + C(fy_year)'
m = smf.ols(formula, data=d).fit(cov_type='HC1')
mc = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['g']})
term = 'nm_border_70mi:post'
pct_change = (np.exp(m.params[term]) - 1) * 100
print(f"n={int(m.nobs)} R2={m.rsquared:.3f}")
print(f"{term}: b={m.params[term]:+.4f}  se(HC1)={m.bse[term]:.4f} p={m.pvalues[term]:.4f}  "
      f"| se(county-cl)={mc.bse[term]:.4f} p={mc.pvalues[term]:.4f}")
print(f"Implied change: {pct_change:+.1f}%")
