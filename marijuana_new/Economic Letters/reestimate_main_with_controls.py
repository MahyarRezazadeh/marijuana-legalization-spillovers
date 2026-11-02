# ============================================================
# Re-estimates the HEADLINE switcher/continuous regressions (the
# ones in Table 1) under the controls that resolved the pre-trend
# test in diagnose_pretrends2.py: baseline (2020) overdose-rate
# level (mean-reversion correction), state fixed effects, and
# division x transition fixed effects -- individually and combined
# -- to see how much of the main result survives each fix.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()

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
s['g'] = pd.Categorical(s['state_abbr']).codes
s['log_pop'] = np.log(s['population'])

# baseline (2020) overdose rate level -- computed directly from the
# panel's own (already-deduplicated, via pivot_table aggfunc='first')
# od_2020 column. NOTE: an earlier version of this script re-derived
# od_2020 via a fresh filter on the raw JSON pull without
# deduplicating first; two Virginia counties (51005, 51019) have an
# exact duplicate record in that raw pull, which fanned out into a
# few duplicated rows on merge and slightly biased every row of this
# table (N=3,799 instead of the correct 3,794). Fixed by using the
# panel's own od_2020 column directly, which is already clean.
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)

specs = {
    'baseline (Table 1, no controls)': '{y} ~ {x} + C(transition)',
    '+ baseline overdose level': '{y} ~ {x} + baseline_rate + C(transition)',
    '+ state FE': '{y} ~ {x} + C(transition) + C(state_abbr)',
    '+ division x transition FE': '{y} ~ {x} + C(transition):C(division)',
    '+ baseline level + state FE': '{y} ~ {x} + baseline_rate + C(transition) + C(state_abbr)',
    '+ baseline level + division x transition FE': '{y} ~ {x} + baseline_rate + C(transition):C(division)',
    '+ baseline level + state x transition FE': '{y} ~ {x} + baseline_rate + C(transition):C(state_abbr)',
}

rows = []
for spec_name, tmpl in specs.items():
    for xname in ['switcher', 'd_dist_mi']:
        sub = s.dropna(subset=['baseline_rate', 'division'])
        formula = tmpl.format(y='d_od', x=xname)
        m = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
        rows.append(dict(spec=spec_name, x=xname, coef=m.params[xname], se=m.bse[xname],
                          p=m.pvalues[xname], n=int(m.nobs)))

res = pd.DataFrame(rows)
pd.set_option('display.width', 160)
for xname in ['switcher', 'd_dist_mi']:
    print(f"\n=== {xname} ===")
    print(res[res['x'] == xname][['spec', 'coef', 'se', 'p', 'n']].to_string(index=False))

res.to_csv(os.path.join(DATA, 'main_with_controls.csv'), index=False)
print("\nSaved main_with_controls.csv")
