import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TAB = os.path.join(BASE, 'tables')

panel = pd.read_csv(os.path.join(DATA, 'partD_county_panel.csv'))

n_sw = int(panel[panel.switcher == 1]['fips5'].nunique())
n_ct = int(panel[panel.switcher == 0]['fips5'].nunique())

rows = [
    {'Variable': 'Home value, ZHVI (2020, median)', 'Value': f"${panel['zhvi_2020'].median():,.0f}"},
    {'Variable': 'Home value, ZHVI (2023, median)', 'Value': f"${panel['zhvi_2023'].median():,.0f}"},
    {'Variable': 'Rent, ZORI (2020, median)', 'Value': f"${panel['zori_2020'].median():,.0f}"},
    {'Variable': 'Rent, ZORI (2023, median)', 'Value': f"${panel['zori_2023'].median():,.0f}"},
    {'Variable': 'Overdose rate per 100k (2020, mean)', 'Value': f"{panel['od_rate20'].mean():.2f}"},
    {'Variable': 'Overdose rate per 100k (2023, mean)', 'Value': f"{panel['od_rate23'].mean():.2f}"},
    {'Variable': 'Switcher counties (N)', 'Value': n_sw},
    {'Variable': 'Control counties (N)', 'Value': n_ct},
    {'Variable': 'Counties with ZHVI coverage (N)', 'Value': int(panel['d_zhvi_log'].notna().sum())},
    {'Variable': 'Counties with ZORI coverage (N)', 'Value': int(panel['d_zori_log'].notna().sum())},
    {'Variable': 'Counties with non-suppressed overdose data (N)', 'Value': int(panel['d_od'].notna().sum())},
]
pd.DataFrame(rows).to_csv(os.path.join(TAB, 'tD0_summary.csv'), index=False)

res = pd.read_csv(os.path.join(TAB, 'tD1_regressions.csv'))
print(res.to_string(index=False))
print("\nSaved tD0_summary.csv")
