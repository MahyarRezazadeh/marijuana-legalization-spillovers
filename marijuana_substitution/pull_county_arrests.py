# ============================================================
# PART A: County-level marijuana vs alcohol arrest data for
# switcher/control counties in the 32 never-legalizing states.
# Concurrent (ThreadPoolExecutor) to keep wall-clock time reasonable.
# ============================================================
import os, time, json, csv, threading
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

with open(os.path.join(PARENT, 'marijuana_new', 'data', '.fbi_api_key.txt')) as f:
    KEY = f.read().strip()
API = "https://api.usa.gov/crime/fbi/cde"

NEVER_LEGAL = ['AL','AR','DC','DE','FL','GA','HI','IA','ID','IN','KS','KY','LA','ME',
               'MN','MS','NC','ND','NE','NH','OH','OK','PA','SC','SD','TN','TX','UT',
               'VA','WI','WV','WY']

MJ_CODES  = "153,158"
ALC_CODES = "270,280"

OUT_AGENCIES = os.path.join(DATA, 'agencies_raw.csv')
OUT_ARRESTS  = os.path.join(DATA, 'county_arrests_raw.csv')
LOG = os.path.join(DATA, 'pull_progress.log')
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

# ── Step 1: agency lists ────────────────────────────────────────
if not os.path.exists(OUT_AGENCIES):
    log("Fetching agency lists for 32 states...")
    rows = []
    for st in NEVER_LEGAL:
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
        w = csv.DictWriter(f, fieldnames=['ori','state_abbr','counties_raw','agency_name','is_nibrs'])
        w.writeheader(); w.writerows(rows)
    log(f"Total agencies: {len(rows)}")
else:
    log("Agency list cache found, skipping fetch.")

import pandas as pd
agencies = pd.read_csv(OUT_AGENCIES)
log(f"Loaded {len(agencies)} agencies.")

done_oris = set()
if os.path.exists(OUT_ARRESTS):
    prev = pd.read_csv(OUT_ARRESTS)
    done_oris = set(prev['ori'].astype(str))
    log(f"Resuming: {len(done_oris)} already done.")
else:
    with open(OUT_ARRESTS, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['ori','mj_2020','mj_2023','alc_2020','alc_2023','status'])

todo = [r for _, r in agencies.iterrows() if str(r['ori']) not in done_oris]
log(f"To process: {len(todo)}")

def sum_year(series, year_suffix):
    return sum(v for k, v in series.items() if k.endswith(year_suffix) and v is not None)

def process_one(row):
    ori = str(row['ori'])
    expected_key = f"{row['agency_name']} Arrests"
    mj_data = get_json(f"/arrest/agency/{ori}/{MJ_CODES}", "from=01-2020&to=12-2023&type=counts")
    alc_data = get_json(f"/arrest/agency/{ori}/{ALC_CODES}", "from=01-2020&to=12-2023&type=counts")
    if mj_data is None or alc_data is None:
        return [ori, '', '', '', '', 'error']
    mj_act = (mj_data.get('actuals') or {}).get(expected_key)
    alc_act = (alc_data.get('actuals') or {}).get(expected_key)
    if mj_act is None or alc_act is None:
        return [ori, '', '', '', '', 'no_exact_match']
    return [ori, sum_year(mj_act, '-2020'), sum_year(mj_act, '-2023'),
             sum_year(alc_act, '-2020'), sum_year(alc_act, '-2023'), 'ok']

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
