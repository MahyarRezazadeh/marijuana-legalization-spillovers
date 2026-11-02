# ============================================================
# Journal of Health Economics draft -- supplementary empirical
# support for the "golden policy window" argument (Section 4):
# pulls county-level HARD-DRUG possession arrest data (FBI CDE
# codes 157 = opium/cocaine/derivatives possession, 159 = synthetic
# narcotics possession -- the two categories closest to overdose-
# relevant drugs, as opposed to 153/158 marijuana codes used
# elsewhere in this project) for the current 31-state never-legal
# universe, 2020-2025, to test whether hard-drug arrests show the
# same transitory (declines at switch, reverts after) pattern as
# overdose mortality -- which would support reading the switch-year
# mortality effect as a real, temporary behavioral shift rather than
# noise.
#
# Reuses the working agency-pull pattern from
# ..\..\marijuana_substitution\pull_county_arrests.py (same API,
# same FBI_API_KEY file, same one-call-per-agency-per-offense-code
# efficiency: each call returns the full monthly series for the
# requested range, so this is NOT 6x slower than the original
# 2-period pull -- still 2 calls/agency, just a longer per-call
# response).
# ============================================================
import os, time, json, csv, threading
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

KEY_PATH = os.path.join(BASE, '..', 'data', '.fbi_api_key.txt')
with open(KEY_PATH) as f:
    KEY = f.read().strip()
API = "https://api.usa.gov/crime/fbi/cde"

dist = pd.read_csv(os.path.join(BASE, '..', 'Economic Letters', 'data', 'distance_by_year_rec_full.csv'))
STATES = sorted(dist['state_abbr'].unique())
print(f"{len(STATES)} states: {STATES}")

OPIOID_COCAINE = "157"   # Drug Possession -- Opium or Cocaine or Their Derivatives
SYNTH_NARCOTIC = "159"   # Drug Possession -- Synthetic Narcotics (fentanyl-class)

OUT_AGENCIES = os.path.join(DATA, 'hd_agencies_raw.csv')
OUT_ARRESTS = os.path.join(DATA, 'hd_county_arrests_raw.csv')
LOG = os.path.join(DATA, 'hd_pull_progress.log')
log_lock = threading.Lock()
write_lock = threading.Lock()

def log(msg):
    with log_lock:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)

def get_json(path, params, retries=4):
    url = f"{API}{path}?{params}&api_key={KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(1 * (attempt + 1))
    return None

# Step 1: agency lists for the current 31-state universe
if not os.path.exists(OUT_AGENCIES):
    log("Fetching agency lists...")
    rows = []
    for st in STATES:
        data = get_json(f"/agency/byStateAbbr/{st}", "")
        if not data:
            log(f"  {st}: FAILED")
            continue
        n = 0
        for county_key, agencies in data.items():
            for a in agencies:
                rows.append({
                    'ori': a.get('ori'), 'state_abbr': a.get('state_abbr', st),
                    'counties_raw': a.get('counties', ''), 'agency_name': a.get('agency_name', ''),
                    'is_nibrs': a.get('is_nibrs', False),
                })
                n += 1
        log(f"  {st}: {n} agencies")
        time.sleep(0.1)
    with open(OUT_AGENCIES, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['ori', 'state_abbr', 'counties_raw', 'agency_name', 'is_nibrs'])
        w.writeheader(); w.writerows(rows)
    log(f"Total agencies: {len(rows)}")
else:
    log("Agency list cache found, skipping fetch.")

agencies = pd.read_csv(OUT_AGENCIES)
log(f"Loaded {len(agencies)} agencies.")

done_oris = set()
if os.path.exists(OUT_ARRESTS):
    prev = pd.read_csv(OUT_ARRESTS)
    done_oris = set(prev['ori'].astype(str))
    log(f"Resuming: {len(done_oris)} already done.")
else:
    with open(OUT_ARRESTS, 'w', newline='', encoding='utf-8') as f:
        header = ['ori'] + [f'op_{y}' for y in range(2020, 2026)] + [f'syn_{y}' for y in range(2020, 2026)] + ['status']
        csv.writer(f).writerow(header)

todo = [r for _, r in agencies.iterrows() if str(r['ori']) not in done_oris]
log(f"To process: {len(todo)}")

def sum_year(series, year):
    suffix = f'-{year}'
    return sum(v for k, v in series.items() if k.endswith(suffix) and v is not None)

def process_one(row):
    ori = str(row['ori'])
    expected_key = f"{row['agency_name']} Arrests"
    op_data = get_json(f"/arrest/agency/{ori}/{OPIOID_COCAINE}", "from=01-2020&to=12-2025&type=counts")
    syn_data = get_json(f"/arrest/agency/{ori}/{SYNTH_NARCOTIC}", "from=01-2020&to=12-2025&type=counts")
    if op_data is None or syn_data is None:
        return [ori] + [''] * 12 + ['error']
    op_act = (op_data.get('actuals') or {}).get(expected_key)
    syn_act = (syn_data.get('actuals') or {}).get(expected_key)
    if op_act is None or syn_act is None:
        return [ori] + [''] * 12 + ['no_exact_match']
    op_vals = [sum_year(op_act, y) for y in range(2020, 2026)]
    syn_vals = [sum_year(syn_act, y) for y in range(2020, 2026)]
    return [ori] + op_vals + syn_vals + ['ok']

n_done = 0
t0 = time.time()
out_f = open(OUT_ARRESTS, 'a', newline='', encoding='utf-8')
writer = csv.writer(out_f)
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(process_one, row): row for row in todo}
    for fut in as_completed(futures):
        result = fut.result()
        with write_lock:
            writer.writerow(result)
            n_done += 1
            if n_done % 200 == 0:
                out_f.flush()
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(todo) - n_done) / rate if rate > 0 else float('inf')
                log(f"  {n_done}/{len(todo)} done, {rate:.2f}/s, ~{remaining/60:.1f} min remaining")
out_f.close()
log("PULL COMPLETE.")
