import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_med_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['g'] = pd.Categorical(s['state_abbr']).codes

dist = pd.read_csv(os.path.join(DATA, 'distance_by_year_med.csv'), dtype={'fips5': str})
with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od2020 = od_raw[od_raw['year'] == 2020][['fips5', 'provisional_drug_overdose']].rename(columns={'provisional_drug_overdose': 'od_2020'})
base = dist[['fips5', 'population']].merge(od2020, on='fips5', how='left')
base['baseline_rate'] = base['od_2020'] / (base['population'] / 1e5)
s = s.merge(base[['fips5', 'baseline_rate']], on='fips5', how='left')

for xname in ['switcher', 'd_dist_mi']:
    sub = s.dropna(subset=['baseline_rate'])
    m = smf.ols(f'd_od ~ {xname} + baseline_rate + C(transition)', data=sub).fit(
        cov_type='cluster', cov_kwds={'groups': sub['g']})
    print(f"{xname}: b={m.params[xname]:+.5f} se={m.bse[xname]:.5f} p={m.pvalues[xname]:.4f} n={int(m.nobs)}")
