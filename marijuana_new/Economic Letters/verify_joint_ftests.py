# ============================================================
# VERIFICATION SCRIPT: the joint F-tests on the two leads
# (lead2plus=0, lead1=0) under division-x-transition FE and under
# baseline-level control were described in main.tex Section 3.2
# but I need to confirm they were actually computed correctly --
# diagnose_pretrends2.py only captured individual coefficient
# p-values, not joint F-tests, for those two specific rows. This
# script computes them properly on the correct event-study panel.
# ============================================================
import os
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

stacked = pd.read_csv(os.path.join(DATA, 'pretrends_panel_rec.csv'), dtype={'fips5': str})

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
stacked['division'] = stacked['state_abbr'].map(DIVISION)
stacked['g'] = pd.Categorical(stacked['state_abbr']).codes

dist = pd.read_csv(os.path.join(DATA, 'distance_by_year_rec.csv'), dtype={'fips5': str})
import json
with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od2020 = od_raw[od_raw['year'] == 2020][['fips5', 'provisional_drug_overdose']].rename(columns={'provisional_drug_overdose': 'od_2020'})
base = dist[['fips5', 'population']].merge(od2020, on='fips5', how='left')
base['baseline_rate'] = base['od_2020'] / (base['population'] / 1e5)
stacked = stacked.merge(base[['fips5', 'baseline_rate']], on='fips5', how='left')

stacked['log_pop'] = None
import numpy as np
base2 = dist[['fips5', 'population']].copy()
base2['log_pop'] = np.log(base2['population'])
stacked = stacked.drop(columns=['log_pop']).merge(base2[['fips5', 'log_pop']], on='fips5', how='left')

# Reproduce diagnose_pretrends2.py's EXACT harmonized sample (dropna on all
# three of baseline_rate, log_pop, division together) so the joint F-tests
# are computed on the identical sample as the individual coefficients I
# already reported in the paper.
harmonized = stacked.dropna(subset=['baseline_rate', 'log_pop', 'division'])
print(f"Harmonized sample size: {len(harmonized)}")

print("\n=== Division x transition FE (harmonized sample) ===")
m1 = smf.ols('d_od ~ lead2plus + lead1 + switch0 + C(transition):C(division)', data=harmonized).fit(
    cov_type='cluster', cov_kwds={'groups': harmonized['g']})
for t in ['lead2plus', 'lead1', 'switch0']:
    print(f"  {t}: b={m1.params[t]:+.4f} p={m1.pvalues[t]:.4f}")
f1 = m1.f_test('lead2plus = 0, lead1 = 0')
print(f"  JOINT F-test (lead2plus=0, lead1=0): F={float(f1.fvalue):.4f} p={float(f1.pvalue):.4f}")

print("\n=== Baseline overdose-level control (harmonized sample) ===")
m2 = smf.ols('d_od ~ lead2plus + lead1 + switch0 + baseline_rate + C(transition)', data=harmonized).fit(
    cov_type='cluster', cov_kwds={'groups': harmonized['g']})
for t in ['lead2plus', 'lead1', 'switch0']:
    print(f"  {t}: b={m2.params[t]:+.4f} p={m2.pvalues[t]:.4f}")
f2 = m2.f_test('lead2plus = 0, lead1 = 0')
print(f"  JOINT F-test (lead2plus=0, lead1=0): F={float(f2.fvalue):.4f} p={float(f2.pvalue):.4f}")

print("\n=== Cross-check: division x transition FE, NOT harmonized (division-only dropna) ===")
sub1b = stacked.dropna(subset=['division'])
m1b = smf.ols('d_od ~ lead2plus + lead1 + switch0 + C(transition):C(division)', data=sub1b).fit(
    cov_type='cluster', cov_kwds={'groups': sub1b['g']})
for t in ['lead2plus', 'lead1', 'switch0']:
    print(f"  {t}: b={m1b.params[t]:+.4f} p={m1b.pvalues[t]:.4f}  (n={int(m1b.nobs)})")
