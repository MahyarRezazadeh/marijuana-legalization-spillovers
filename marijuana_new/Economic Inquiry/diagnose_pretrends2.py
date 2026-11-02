# ============================================================
# Exhaustive attempt to explain away the pre-trend found in
# analyze_pretrends.py. Tries, one at a time and combined:
#   - state fixed effects (not just 4-region)
#   - Census division (9-region) x transition fixed effects
#   - baseline (2020) overdose-rate LEVEL as a covariate (tests
#     whether this is simple mean-reversion: counties with high
#     initial rates mechanically decline more)
#   - log population as a covariate (rate volatility / rurality)
#   - state x transition fixed effects (most granular)
# Reports lead1/lead2plus/switch0 under each spec so we can see
# whether any control actually closes the gap, or whether the
# pre-trend is robust to controls (i.e., a real problem).
# ============================================================
import os, json
import numpy as np
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

# baseline (2020) overdose rate level, for the mean-reversion test
dist = pd.read_csv(os.path.join(DATA, 'distance_by_year_rec.csv'), dtype={'fips5': str})
with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od2020 = od_raw[od_raw['year'] == 2020][['fips5', 'provisional_drug_overdose']].rename(
    columns={'provisional_drug_overdose': 'od_2020'})
base = dist[['fips5', 'population']].merge(od2020, on='fips5', how='left')
base['baseline_rate'] = base['od_2020'] / (base['population'] / 1e5)
base['log_pop'] = np.log(base['population'])
stacked = stacked.merge(base[['fips5', 'baseline_rate', 'log_pop']], on='fips5', how='left')

specs = {
    'baseline (no controls)': 'd_od ~ lead2plus + lead1 + switch0 + C(transition)',
    'state FE': 'd_od ~ lead2plus + lead1 + switch0 + C(transition) + C(state_abbr)',
    'division x transition FE': 'd_od ~ lead2plus + lead1 + switch0 + C(transition):C(division)',
    'state x transition FE': 'd_od ~ lead2plus + lead1 + switch0 + C(transition):C(state_abbr)',
    'baseline overdose LEVEL control': 'd_od ~ lead2plus + lead1 + switch0 + baseline_rate + C(transition)',
    'log population control': 'd_od ~ lead2plus + lead1 + switch0 + log_pop + C(transition)',
    'baseline level + log pop': 'd_od ~ lead2plus + lead1 + switch0 + baseline_rate + log_pop + C(transition)',
    'state FE + baseline level': 'd_od ~ lead2plus + lead1 + switch0 + baseline_rate + C(transition) + C(state_abbr)',
    'division x transition + baseline level': 'd_od ~ lead2plus + lead1 + switch0 + baseline_rate + C(transition):C(division)',
}

results = []
for name, formula in specs.items():
    sub = stacked.dropna(subset=['baseline_rate', 'log_pop', 'division'])
    try:
        m = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
        row = dict(spec=name, n=int(m.nobs))
        for term in ['lead2plus', 'lead1', 'switch0']:
            row[f'{term}_b'] = m.params.get(term, np.nan)
            row[f'{term}_p'] = m.pvalues.get(term, np.nan)
        results.append(row)
    except Exception as e:
        print(f"{name}: FAILED ({e})")

res = pd.DataFrame(results)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)
print(res.round(4).to_string(index=False))
res.to_csv(os.path.join(DATA, 'pretrends_diagnostics.csv'), index=False)
print("\nSaved pretrends_diagnostics.csv")
