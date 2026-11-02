import os, json
import urllib.request
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MARIJUANA_NEW = os.path.dirname(BASE)
ROOT = os.path.dirname(MARIJUANA_NEW)
DATA = os.path.join(BASE, 'data')
HOUSING_DATA = os.path.join(ROOT, 'marijuana_housing_overdose', 'data')

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['effective_date', 'retail_date'])
states_needed = set(legal['state_abbr'])

url = "https://data.cdc.gov/resource/xkb8-kh2a.json?indicator=Number%20of%20Drug%20Overdose%20Deaths&%24limit=50000"
with urllib.request.urlopen(url, timeout=60) as resp:
    raw = json.loads(resp.read().decode())
df = pd.DataFrame(raw)
df = df[df['state'].isin(states_needed)].copy()
df['data_value'] = pd.to_numeric(df['data_value'], errors='coerce')
df['date'] = pd.to_datetime(df['month'] + ' ' + df['year'], format='%B %Y')
print(f"Pulled {len(df)} rows for {df['state'].nunique()} states (need {len(states_needed)}), "
      f"{df['date'].min().date()} to {df['date'].max().date()}")
missing = states_needed - set(df['state'].unique())
if missing:
    print(f"WARNING missing states: {missing}")

pop = pd.read_csv(os.path.join(HOUSING_DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop_state = pop[pop['COUNTY'] == 0][['STNAME', 'POPESTIMATE2020']]
STATE_ABBR = dict(zip(legal['state'], legal['state_abbr']))
pop_state['state'] = pop_state['STNAME'].map(STATE_ABBR)
pop_state = pop_state.dropna(subset=['state'])[['state', 'POPESTIMATE2020']]

df = df.merge(pop_state, on='state', how='left')
df['rate_per_100k'] = df['data_value'] / (df['POPESTIMATE2020'] / 1e5)
df = df.merge(legal[['state_abbr', 'effective_date', 'retail_date']], left_on='state', right_on='state_abbr', how='left')

df.to_csv(os.path.join(DATA, 'state_overdose_trend_new.csv'), index=False)
print(f"Saved state_overdose_trend_new.csv ({len(df)} rows)")
