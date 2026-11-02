# ============================================================
# Pulls the long-run state-level monthly (12-month-ending) drug
# overdose death series (CDC VSRR, 2015-present) for the 21
# states that have legalized recreational marijuana, for a
# trend visualization with medical/recreational legalization
# dates marked -- descriptive only, not part of the regression
# design.
# ============================================================
import os, json
import urllib.request
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

# Recreational legalization year (already used elsewhere in this project) and month/day
# where known; medical marijuana legalization year (general knowledge -- widely documented,
# stable public facts, not freshly sourced this session).
LEGAL_DATES = {
    # state: (medical_date, recreational_date)  "YYYY-MM"
    'WA': ('1998-11', '2012-12'), 'CO': ('2000-11', '2012-12'), 'OR': ('1998-11', '2015-07'),
    'AK': ('1998-11', '2015-02'), 'NV': ('2000-11', '2017-01'), 'MA': ('2012-11', '2018-12'),
    'CA': ('1996-11', '2018-01'), 'MI': ('2008-11', '2019-12'), 'IL': ('2013-08', '2020-01'),
    'AZ': ('2010-11', '2021-01'), 'RI': ('2006-01', '2022-05'), 'VT': ('2004-06', '2022-07'),
    'MT': ('2004-11', '2022-01'), 'NJ': ('2010-01', '2022-04'), 'NM': ('2007-04', '2022-04'),
    'NY': ('2014-07', '2022-03'), 'CT': ('2012-06', '2023-07'), 'MD': ('2014-04', '2023-07'),
    'MO': ('2018-12', '2023-02'), 'MN': ('2014-05', '2023-08'), 'OH': ('2016-09', '2023-08'),
}

url = "https://data.cdc.gov/resource/xkb8-kh2a.json?indicator=Number%20of%20Drug%20Overdose%20Deaths&%24limit=50000"
with urllib.request.urlopen(url, timeout=60) as resp:
    raw = json.loads(resp.read().decode())
df = pd.DataFrame(raw)
df = df[df['state'].isin(LEGAL_DATES.keys())].copy()
df['data_value'] = pd.to_numeric(df['data_value'], errors='coerce')
df['date'] = pd.to_datetime(df['month'] + ' ' + df['year'], format='%B %Y')
print(f"Pulled {len(df)} rows for {df['state'].nunique()} states, "
      f"{df['date'].min().date()} to {df['date'].max().date()}")

# ── State population (2020 vintage, from the already-downloaded county file's state totals) ──
pop = pd.read_csv(os.path.join(DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop_state = pop[pop['COUNTY'] == 0][['STNAME', 'POPESTIMATE2020']]
STATE_ABBR = {
    'Washington':'WA','Colorado':'CO','Oregon':'OR','Alaska':'AK','Nevada':'NV','Massachusetts':'MA',
    'California':'CA','Michigan':'MI','Illinois':'IL','Arizona':'AZ','Rhode Island':'RI','Vermont':'VT',
    'Montana':'MT','New Jersey':'NJ','New Mexico':'NM','New York':'NY','Connecticut':'CT','Maryland':'MD',
    'Missouri':'MO','Minnesota':'MN','Ohio':'OH',
}
pop_state['state'] = pop_state['STNAME'].map(STATE_ABBR)
pop_state = pop_state.dropna(subset=['state'])[['state', 'POPESTIMATE2020']]

df = df.merge(pop_state, on='state', how='left')
df['rate_per_100k'] = df['data_value'] / (df['POPESTIMATE2020'] / 1e5)

df['legal_med'] = df['state'].map(lambda s: LEGAL_DATES[s][0])
df['legal_rec'] = df['state'].map(lambda s: LEGAL_DATES[s][1])

df.to_csv(os.path.join(DATA, 'state_overdose_trend.csv'), index=False)
with open(os.path.join(DATA, 'legal_dates.json'), 'w') as f:
    json.dump(LEGAL_DATES, f, indent=2)
print(f"Saved state_overdose_trend.csv ({len(df)} rows), legal_dates.json")
print(df.groupby('state')['date'].agg(['min','max','count']))
