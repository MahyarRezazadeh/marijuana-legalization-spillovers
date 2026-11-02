import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

new_panel = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel_CORRECTED.csv'), dtype={'fips5': str})
snap = new_panel[new_panel['fy_year'] == 2025].copy()
snap['dispensary_per_1000'] = snap['dispensary_count'] / snap['population'] * 1000

cowa = snap[snap.state_abbr.isin(['CO', 'WA'])].copy()
formula = 'dispensary_per_1000 ~ border_70mi + log_population + C(state_abbr)'
m = smf.ols(formula, data=cowa).fit(cov_type='HC1')
print(f"CO+WA entry density (CORRECTED WA counts): b={m.params['border_70mi']:+.4f} "
      f"p={m.pvalues['border_70mi']:.4f} n={int(m.nobs)}")

wa_only = snap[snap.state_abbr == 'WA']
m2 = smf.ols('dispensary_per_1000 ~ border_70mi + log_population', data=wa_only).fit(cov_type='HC1')
print(f"WA only entry density (CORRECTED): b={m2.params['border_70mi']:+.4f} "
      f"p={m2.pvalues['border_70mi']:.4f} n={int(m2.nobs)}")
