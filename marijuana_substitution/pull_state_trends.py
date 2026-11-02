# ============================================================
# PART B: State-level marijuana vs. alcohol arrest trends,
# before/after each state's OWN recreational legalization date.
# Fast (21 states x 2 offense-groups = 42 calls).
# ============================================================
import os, time, json, urllib.request, urllib.error
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

with open(os.path.join(PARENT, 'marijuana_new', 'data', '.fbi_api_key.txt')) as f:
    KEY = f.read().strip()
API = "https://api.usa.gov/crime/fbi/cde"

LEGAL_STATES = {
    'WA': 2014, 'CO': 2014, 'OR': 2015, 'AK': 2016, 'NV': 2017, 'MA': 2018,
    'CA': 2018, 'MI': 2019, 'IL': 2020, 'AZ': 2021, 'RI': 2022, 'VT': 2022,
    'MT': 2022, 'NJ': 2022, 'NM': 2022, 'NY': 2022, 'CT': 2023, 'MD': 2023,
    'MO': 2023, 'MN': 2024, 'OH': 2024,
}

MJ_CODES  = "153,158"   # Drug Sale/Mfg Marijuana + Drug Possession Marijuana
ALC_CODES = "270,280"   # Liquor Law Violations + Drunkenness

def get_json(path, params, retries=5):
    url = f"{API}{path}?{params}&api_key={KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  HTTPError {e.code} on {path}: {e.read().decode()[:200]}")
            return None
        except Exception as e:
            time.sleep(1 * (attempt + 1))
    return None

rows = []
for st, year in LEGAL_STATES.items():
    for label, codes in [('marijuana', MJ_CODES), ('alcohol', ALC_CODES)]:
        data = get_json(f"/arrest/state/{st}/{codes}", "from=01-2005&to=12-2023&type=counts")
        if data is None:
            print(f"  {st} {label}: FAILED")
            continue
        actuals = data.get('actuals') or {}
        rates = data.get('rates') or {}
        pops = data.get('populations', {})
        pop_total = pops.get('population', {})
        pop_part  = pops.get('participated_population', {})
        # pick the state-specific series (only one key expected for a single-state query)
        if len(actuals) == 1:
            series_key = list(actuals.keys())[0]
            pop_key = series_key.replace(' Arrests', '')  # population dicts omit " Arrests" suffix
            act = actuals[series_key]
            rt = rates.get(series_key, {})
            pt = pop_total.get(pop_key, {})
            pp = pop_part.get(pop_key, {})
        else:
            print(f"  {st} {label}: unexpected multi-series response, keys={list(actuals.keys())}")
            continue
        for month, count in act.items():
            part_pop = pp.get(month)
            tot_pop = pt.get(month)
            adj_rate = (count / part_pop * 100000) if part_pop else None
            coverage = (part_pop / tot_pop * 100) if (part_pop and tot_pop) else None
            rows.append({
                'state_abbr': st, 'rec_year': year, 'offense': label,
                'month': month, 'actual': count, 'rate_per_100k': rt.get(month),
                'rate_per_100k_covadj': adj_rate, 'coverage_pct': coverage,
            })
        print(f"  {st} {label}: {len(act)} months pulled")
    time.sleep(0.15)

df = pd.DataFrame(rows)
out_path = os.path.join(DATA, 'state_trends_mj_alcohol.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows to {out_path}")
print(df.groupby(['state_abbr', 'offense']).size())
