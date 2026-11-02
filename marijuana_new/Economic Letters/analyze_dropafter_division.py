# ============================================================
# Same specification structure as analyze_cumulative_treated.py
# (transition FE + baseline-2020-rate control + Census-division
# FE), but on the STANDARD panel that drops a county from the
# panel after it switches (stacked_panel_rec_full.csv) instead of
# keeping it in as "treated" for subsequent years. This isolates
# just the switch-transition effect itself, with the regional
# control added on top of Table 1's preferred specification.
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

DIVISION = {
    'CT':'NewEngland','ME':'NewEngland','MA':'NewEngland','NH':'NewEngland','RI':'NewEngland','VT':'NewEngland',
    'NJ':'MidAtlantic','NY':'MidAtlantic','PA':'MidAtlantic',
    'IL':'ENCentral','IN':'ENCentral','MI':'ENCentral','OH':'ENCentral','WI':'ENCentral',
    'IA':'WNCentral','KS':'WNCentral','MN':'WNCentral','MO':'WNCentral','NE':'WNCentral','ND':'WNCentral','SD':'WNCentral',
    'DE':'SAtlantic','FL':'SAtlantic','GA':'SAtlantic','MD':'SAtlantic','NC':'SAtlantic','SC':'SAtlantic','VA':'SAtlantic','DC':'SAtlantic','WV':'SAtlantic',
    'AL':'ESCentral','KY':'ESCentral','MS':'ESCentral','TN':'ESCentral',
    'AR':'WSCentral','LA':'WSCentral','OK':'WSCentral','TX':'WSCentral',
    'AZ':'Mountain','CO':'Mountain','ID':'Mountain','MT':'Mountain','NV':'Mountain','NM':'Mountain','UT':'Mountain','WY':'Mountain',
    'AK':'Pacific','CA':'Pacific','HI':'Pacific','OR':'Pacific','WA':'Pacific',
}
s['division'] = s['state_abbr'].map(DIVISION)
s = s.dropna(subset=['baseline_rate', 'division'])
s['g'] = pd.Categorical(s['state_abbr']).codes

print(f"N={len(s)}, switch events={int(s['switcher'].sum())}, control obs={int((s['switcher']==0).sum())}")
print(f"Distinct switcher counties: {s.loc[s['switcher']==1,'fips5'].nunique()}")

m = smf.ols('d_od ~ switcher + baseline_rate + C(transition) + C(division)', data=s).fit(
    cov_type='cluster', cov_kwds={'groups': s['g']})

print(f"\nR-squared: {m.rsquared:.4f}\n")
for term in m.params.index:
    label = term
    if term.startswith('C(transition)'):
        label = 'Transition: ' + term.split("T.")[-1].rstrip(']')
    elif term.startswith('C(division)'):
        label = 'Division: ' + term.split("T.")[-1].rstrip(']')
    elif term == 'switcher':
        label = 'Switcher (drops after switching)'
    elif term == 'baseline_rate':
        label = 'Baseline (2020) overdose rate'
    print(f"{label:50s} b={m.params[term]:+.5f} se={m.bse[term]:.5f} p={m.pvalues[term]:.4f}")

out = {term: dict(coef=float(m.params[term]), se=float(m.bse[term]), p=float(m.pvalues[term])) for term in m.params.index}
with open(os.path.join(DATA, 'dropafter_division_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved dropafter_division_results.json")
