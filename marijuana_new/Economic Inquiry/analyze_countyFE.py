# ============================================================
# County fixed effects robustness check: adds a county fixed
# effect to the preferred (transition FE + baseline-rate control)
# specification. Since the outcome is already a transition-level
# first difference, a county FE here absorbs any county-specific
# average yearly change across its transitions (e.g. a county on
# an idiosyncratically steeper decline every year, for reasons
# unrelated to switching) -- a stricter test than state FE, using
# only within-county variation across its own transitions.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)
s['g'] = pd.Categorical(s['state_abbr']).codes

# only counties appearing in >1 transition contribute to a within-county
# comparison; report how many switchers that excludes for transparency
n_per_county = s.groupby('fips5').size()
print(f"Counties with only 1 transition-row (no within-county variation): {(n_per_county==1).sum()} of {len(n_per_county)}")

specs = {
    'Preferred (transition FE + baseline rate)': 'd_od ~ {x} + baseline_rate + C(transition)',
    '+ County fixed effects': 'd_od ~ {x} + baseline_rate + C(transition) + C(fips5)',
}

rows = []
for spec_name, tmpl in specs.items():
    for xname in ['switcher', 'd_dist_mi']:
        sub = s.dropna(subset=['baseline_rate'])
        formula = tmpl.format(x=xname)
        m = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
        rows.append(dict(spec=spec_name, x=xname, coef=m.params[xname], se=m.bse[xname],
                          p=m.pvalues[xname], n=int(m.nobs)))
        print(f"{spec_name:45s} [{xname:10s}] b={m.params[xname]:+.5f} se={m.bse[xname]:.5f} "
              f"p={m.pvalues[xname]:.4f} n={int(m.nobs)}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(DATA, 'countyFE_results.csv'), index=False)
print("\nSaved countyFE_results.csv")
