import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_med_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['g'] = pd.Categorical(s['state_abbr']).codes

# baseline_rate computed directly from the panel's own (already-
# deduplicated) od_2020 column -- see reestimate_main_with_controls.py
# for why the external-merge approach used here previously could
# double-count two Virginia counties with a duplicate raw record.
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)

out = {}
for xname in ['switcher', 'd_dist_mi']:
    sub = s.dropna(subset=['baseline_rate'])
    m = smf.ols(f'd_od ~ {xname} + baseline_rate + C(transition)', data=sub).fit(
        cov_type='cluster', cov_kwds={'groups': sub['g']})
    print(f"{xname}: b={m.params[xname]:+.5f} se={m.bse[xname]:.5f} p={m.pvalues[xname]:.4f} n={int(m.nobs)}")
    out[xname] = dict(coef=float(m.params[xname]), se=float(m.bse[xname]), p=float(m.pvalues[xname]), n=int(m.nobs))

with open(os.path.join(DATA, 'medical_adjusted_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved medical_adjusted_results.json")
