import os, json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TAB = os.path.join(BASE, 'tables')

partA = pd.read_csv(os.path.join(DATA, 'partA_county_panel.csv'))
with open(os.path.join(DATA, 'partA_results.json')) as f:
    resA = json.load(f)
resB = pd.read_csv(os.path.join(DATA, 'partB_twfe_results.csv'))

# Table A1: summary stats
n_sw = int(partA[partA.switcher == 1]['fips5'].nunique())
n_ct = int(partA[partA.switcher == 0]['fips5'].nunique())
pd.DataFrame({
    'Variable': ['Marijuana arrest rate per 100k (2020)', 'Marijuana arrest rate per 100k (2023)',
                 'Alcohol arrest rate per 100k (2020)', 'Alcohol arrest rate per 100k (2023)',
                 'Switcher counties (N)', 'Control counties (N)', 'Counties matched to FIPS (N)'],
    'Value': [f"{partA['mj_rate20'].mean():.2f}", f"{partA['mj_rate23'].mean():.2f}",
              f"{partA['alc_rate20'].mean():.2f}", f"{partA['alc_rate23'].mean():.2f}",
              n_sw, n_ct, len(partA)]
}).to_csv(os.path.join(TAB, 'tA1_summary.csv'), index=False)

# Table A2: regression results
rows = []
for key, label in [('mj_switcher', 'Marijuana: Switcher'),
                    ('mj_contdist_mi', 'Marijuana: Distance (per 100 mi)'),
                    ('mj_contdist_log', 'Marijuana: Log-distance (robustness)'),
                    ('alc_switcher', 'Alcohol: Switcher'),
                    ('alc_contdist_mi', 'Alcohol: Distance (per 100 mi)'),
                    ('alc_contdist_log', 'Alcohol: Log-distance (robustness)')]:
    b, se, p, n = resA[key]
    rows.append({'Spec': label, 'Coef': b, 'SE': se, 'p': p, 'N': n})
pd.DataFrame(rows).to_csv(os.path.join(TAB, 'tA2_regressions.csv'), index=False)

# Table B1: TWFE
resB.to_csv(os.path.join(TAB, 'tB1_twfe.csv'), index=False)

print("Tables saved.")
print(pd.DataFrame(rows))
