# ============================================================
# Pulls CDC VSRR county-level 12-month-ending drug overdose death
# counts for December of every year 2020-2025 (the full range
# county-level VSRR data is available for), to support the new
# multi-period stacked-difference panel design (replacing the
# earlier two-snapshot Dec-2020/Dec-2023 design).
# ============================================================
import os, json, urllib.request, time

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

def pull_year(year):
    rows = []
    offset = 0
    limit = 5000
    while True:
        url = (f"https://data.cdc.gov/resource/gb4e-yj24.json?year={year}&month=12"
               f"&$limit={limit}&$offset={offset}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            chunk = json.load(r)
        rows.extend(chunk)
        if len(chunk) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return rows

all_rows = []
for yr in YEARS:
    rows = pull_year(yr)
    print(f"{yr}: {len(rows)} county rows")
    all_rows.extend(rows)

with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), 'w', encoding='utf-8') as f:
    json.dump(all_rows, f)
print(f"Saved overdose_county_multiyear_raw.json ({len(all_rows)} total rows)")
