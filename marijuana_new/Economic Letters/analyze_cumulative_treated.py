# ============================================================
# "Effect of being a switcher, from the point of switching onward"
# -- a cumulative/pooled post-treatment indicator, not separate
# lag1/lag2plus dummies (that's the persistence check). Treated_it
# = 1 for the switch transition AND every transition after it for
# that county (kept in the panel rather than dropped, unlike the
# main design); 0 before switching (including eventual switchers'
# pre-switch rows) and always for never-switchers.
#
# Chosen specification: transition FE + baseline-2020-rate control
# (mean reversion) + Census-division FE (the regional-confound fix
# that resolved the 2023-2024 issue) -- NOT state or county FE,
# which we already know destroy power here.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

THRESH = 50
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

dist = pd.read_csv(os.path.join(DATA, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()
dist = dist.merge(od, on='fips5', how='left')
dist = dist[dist['dist_2020'] >= THRESH].copy()  # exclude always-near, as in the main design

def find_switch_year(row):
    for y in YEARS[1:]:
        if row[f'dist_{y-1}'] >= THRESH and row[f'dist_{y}'] < THRESH:
            return y
    return np.nan
dist['switch_year'] = dist.apply(find_switch_year, axis=1)

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
dist['division'] = dist['state_abbr'].map(DIVISION)

rows = []
for t in YEARS[1:]:
    t0 = t - 1
    sub = dist.copy()
    sub['transition'] = f'{t0}-{t}'
    sub['treated'] = (sub['switch_year'].notna() & (t >= sub['switch_year'])).astype(float)
    sub['od_rate_pre'] = sub[f'od_{t0}'] / (sub['population'] / 1e5)
    sub['od_rate_post'] = sub[f'od_{t}'] / (sub['population'] / 1e5)
    sub['d_od'] = sub['od_rate_post'] - sub['od_rate_pre']
    rows.append(sub[['fips5', 'state_abbr', 'division', 'population', 'transition', 'treated', 'd_od']])

stacked = pd.concat(rows, ignore_index=True)
stacked = stacked[stacked['d_od'].notna() & (stacked['population'] > 0)].copy()

od2020 = od_raw[od_raw['year'] == 2020][['fips5', 'provisional_drug_overdose']].rename(columns={'provisional_drug_overdose': 'od_2020'})
base = dist[['fips5', 'population']].merge(od2020, on='fips5', how='left')
base['baseline_rate'] = base['od_2020'] / (base['population'] / 1e5)
stacked = stacked.merge(base[['fips5', 'baseline_rate']], on='fips5', how='left')
stacked = stacked.dropna(subset=['baseline_rate', 'division'])
stacked['g'] = pd.Categorical(stacked['state_abbr']).codes

print(f"N={len(stacked)}, treated obs={int(stacked['treated'].sum())}, control obs={int((stacked['treated']==0).sum())}")
print(f"Distinct counties ever treated: {stacked.loc[stacked['treated']==1,'fips5'].nunique()}")

m = smf.ols('d_od ~ treated + baseline_rate + C(transition) + C(division)', data=stacked).fit(
    cov_type='cluster', cov_kwds={'groups': stacked['g']})

print(f"\nR-squared: {m.rsquared:.4f}\n")
for term in m.params.index:
    label = term
    if term.startswith('C(transition)'):
        label = 'Transition: ' + term.split("T.")[-1].rstrip(']')
    elif term.startswith('C(division)'):
        label = 'Division: ' + term.split("T.")[-1].rstrip(']')
    elif term == 'treated':
        label = 'Treated (switcher, from switch year onward)'
    elif term == 'baseline_rate':
        label = 'Baseline (2020) overdose rate'
    print(f"{label:50s} b={m.params[term]:+.5f} se={m.bse[term]:.5f} p={m.pvalues[term]:.4f}")

out = {term: dict(coef=float(m.params[term]), se=float(m.bse[term]), p=float(m.pvalues[term])) for term in m.params.index}
with open(os.path.join(DATA, 'cumulative_treated_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved cumulative_treated_results.json")
