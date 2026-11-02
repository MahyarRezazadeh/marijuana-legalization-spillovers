# ============================================================
# Pulls County Business Patterns (CBP) establishment counts for
# five ancillary/non-plant-touching NAICS sectors plausibly linked
# to serving a neighboring legal cannabis market (the B1 mechanism):
#   561621 - Security systems services (except locksmiths)
#   238220 - Plumbing, heating, and air-conditioning contractors
#   323111 - Commercial printing (packaging/labels)
#   5416   - Management, scientific & technical consulting services
#   541380 - Testing laboratories
# for every county in the project's 31-never-legal-state universe,
# years 2020-2023 (CBP's most recent available reference year as of
# this pull is 2023; 2024/2025 not yet released).
# ============================================================
import os
import time
import requests
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
PARENT_DATA = os.path.join(BASE, '..', 'data')

with open(os.path.join(PARENT_DATA, 'census_api_key.txt')) as f:
    KEY = f.read().strip()

YEARS = [2020, 2021, 2022, 2023]
NAICS_CODES = {
    '561621': 'security_systems',
    '238220': 'hvac_contractors',
    '323111': 'commercial_printing',
    '5416': 'mgmt_sci_tech_consulting',
    '541380': 'testing_labs',
}

STATE_FIPS = {
    'AL': '01', 'AR': '05', 'DC': '11', 'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15',
    'IA': '19', 'ID': '16', 'IN': '18', 'KS': '20', 'KY': '21', 'LA': '22', 'MN': '27',
    'MS': '28', 'NC': '37', 'ND': '38', 'NE': '31', 'NH': '33', 'OH': '39', 'OK': '40',
    'PA': '42', 'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VA': '51',
    'WI': '55', 'WV': '54', 'WY': '56',
}

dist = pd.read_csv(os.path.join(PARENT_DATA, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
universe_states = sorted(dist['state_abbr'].unique())
assert set(universe_states) == set(STATE_FIPS.keys()), (set(universe_states) ^ set(STATE_FIPS.keys()))

session = requests.Session()
all_rows = []
n_calls = 0
for year in YEARS:
    for naics, label in NAICS_CODES.items():
        for state_abbr, state_fips in STATE_FIPS.items():
            url = (f"https://api.census.gov/data/{year}/cbp"
                   f"?get=ESTAB&for=county:*&in=state:{state_fips}&NAICS2017={naics}&key={KEY}")
            r = session.get(url, timeout=30)
            n_calls += 1
            if r.status_code != 200:
                print(f"  FAIL {year} {label} {state_abbr}: HTTP {r.status_code}")
                continue
            data = r.json()
            cols = data[0]
            for row in data[1:]:
                d = dict(zip(cols, row))
                all_rows.append({
                    'year': year, 'sector': label, 'naics': naics,
                    'fips5': d['state'] + d['county'],
                    'estab': int(d['ESTAB']),
                })
        print(f"  done: {year} {label} ({n_calls} calls so far)")

raw = pd.DataFrame(all_rows)
raw.to_csv(os.path.join(DATA, 'cbp_ancillary_raw.csv'), index=False)
print(f"\nTotal API calls: {n_calls}")
print(f"Total rows: {len(raw)}")
print(raw.groupby(['sector', 'year'])['estab'].agg(['count', 'sum']))
