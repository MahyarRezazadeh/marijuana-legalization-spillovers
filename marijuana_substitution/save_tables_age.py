import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TAB = os.path.join(BASE, 'tables')

panel = pd.read_csv(os.path.join(DATA, 'partC_age_panel.csv'))
AGE_ORDER = ['<21', '21-30', '31-45', '46+']

# Table C0: summary stats by age bin (rates, N)
rows = []
for age_bin in AGE_ORDER:
    s = panel[panel['age_bin'] == age_bin]
    rows.append({
        'Age bin': age_bin,
        'Marijuana rate 2020': round(s['mj_rate20'].mean(), 3),
        'Marijuana rate 2023': round(s['mj_rate23'].mean(), 3),
        'Alcohol rate 2020': round(s['alc_rate20'].mean(), 3),
        'Alcohol rate 2023': round(s['alc_rate23'].mean(), 3),
        'N counties': len(s),
    })
pd.DataFrame(rows).to_csv(os.path.join(TAB, 'tC0_age_summary.csv'), index=False)

# Table C1 (pretty): regression results, primary dist_100mi + switcher specs only
res = pd.read_csv(os.path.join(TAB, 'tC1_age_regressions.csv'))
res['age_bin'] = pd.Categorical(res['age_bin'], categories=AGE_ORDER, ordered=True)
pretty = res[res['spec'].isin(['switcher', 'dist_100mi'])].sort_values(['age_bin', 'outcome', 'spec'])
pretty = pretty[['age_bin', 'outcome', 'spec', 'coef', 'se', 'p', 'n', 'n_switchers', 'n_controls']]
pretty.to_csv(os.path.join(TAB, 'tC1_pretty.csv'), index=False)

print("Saved tC0_age_summary.csv and tC1_pretty.csv")
print(pd.DataFrame(rows))
print()
print(pretty.to_string(index=False))
