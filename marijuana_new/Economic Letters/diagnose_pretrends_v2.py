# ============================================================
# Division x transition FE check, on the UNIFIED pre-trends panel
# (pretrends_panel_rec_v2.csv, built directly from the same panel
# as Table 1/2), for consistency with analyze_pretrends_v2.py.
# ============================================================
import os, json
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'pretrends_panel_rec_v2.csv'), dtype={'fips5': str})

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

# Harmonize to the SAME sample as the baseline-rate-adjusted regression
# (non-missing baseline_rate), so all rows of the eventual Table 2 are
# on one consistent N, exactly as the table note claims.
sub = s.dropna(subset=['baseline_rate', 'division'])
print(f"Harmonized sample size: {len(sub)}")

m = smf.ols('d_od ~ lead2plus + lead1 + switch0 + C(transition):C(division)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['g']})
for t in ['lead2plus', 'lead1', 'switch0']:
    print(f"  {t}: b={m.params[t]:+.4f} p={m.pvalues[t]:.4f}")
f = m.f_test('lead2plus = 0, lead1 = 0')
print(f"  JOINT F-test (leads=0): F={float(f.fvalue):.4f} p={float(f.pvalue):.4f}")

out = dict(n=len(sub),
           lead2plus=dict(coef=float(m.params['lead2plus']), p=float(m.pvalues['lead2plus'])),
           lead1=dict(coef=float(m.params['lead1']), p=float(m.pvalues['lead1'])),
           switch0=dict(coef=float(m.params['switch0']), p=float(m.pvalues['switch0'])),
           joint_f=dict(F=float(f.fvalue), p=float(f.pvalue)))
with open(os.path.join(DATA, 'pretrends_division_v2.json'), 'w') as fh:
    json.dump(out, fh, indent=2)
print("Saved pretrends_division_v2.json")
