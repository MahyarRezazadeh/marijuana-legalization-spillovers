# ============================================================
# Full coefficient report for the binary (switcher) specification
# with county fixed effects added -- not just the switcher
# coefficient (as in Table 2's "+ County fixed effects" row), but
# every other estimated coefficient too: the baseline-2020-rate
# control and each transition (period) dummy. County fixed effects
# themselves (~1,000+ individual county dummies) are estimated but
# not individually listed, matching standard practice -- reporting
# every unit fixed effect in a table is neither standard nor useful.
#
# Panel: the same stacked_panel_rec_full.csv used throughout, which
# already drops a county from every transition AFTER it switches
# (built into build_multiperiod_panel.py's "still_at_risk" filter) --
# so "drop the switcher after they become a switcher" is already the
# panel's construction rule, not a new restriction to add.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)  # initial (2020) overdose rate
s = s.dropna(subset=['baseline_rate'])
s['g'] = pd.Categorical(s['state_abbr']).codes

TRANSITIONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
s['transition'] = pd.Categorical(s['transition'], categories=TRANSITIONS, ordered=True)

m = smf.ols('d_od ~ switcher + baseline_rate + C(transition) + C(fips5)', data=s).fit(
    cov_type='cluster', cov_kwds={'groups': s['g']})

print(f"N = {int(m.nobs)}, counties (county FE groups) = {s['fips5'].nunique()}")
print(f"R-squared (mostly mechanical, from ~{s['fips5'].nunique()} county dummies): {m.rsquared:.4f}\n")

rows = []
report_terms = ['Intercept', 'switcher', 'baseline_rate'] + [t for t in m.params.index if t.startswith('C(transition)')]
for term in report_terms:
    if term not in m.params.index:
        continue
    label = term
    if term.startswith('C(transition)'):
        label = 'Transition: ' + term.split("T.")[-1].rstrip(']')
    elif term == 'switcher':
        label = 'Switcher (coefficient of interest)'
    elif term == 'baseline_rate':
        label = 'Baseline (2020) overdose rate'
    rows.append(dict(term=label, coef=m.params[term], se=m.bse[term], p=m.pvalues[term]))
    print(f"{label:45s} b={m.params[term]:+.5f} se={m.bse[term]:.5f} p={m.pvalues[term]:.4f}")

n_county_dummies = sum(1 for t in m.params.index if t.startswith('C(fips5)'))
print(f"\n[{n_county_dummies} county fixed-effect dummies estimated, not individually listed]")

pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'countyFE_full_coefficients.csv'), index=False)
print("\nSaved countyFE_full_coefficients.csv")
