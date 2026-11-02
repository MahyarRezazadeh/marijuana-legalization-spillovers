# ============================================================
# Mechanism check #3 (supply-side market disruption): same pull as
# pull_harddrug_arrests.py but SALE/MANUFACTURING codes (152 =
# opium/cocaine/derivatives sale-mfg, 154 = synthetic narcotics
# sale-mfg) instead of possession -- tests whether a new legal
# market temporarily disrupts local illicit dealer networks (who
# often sell marijuana alongside harder drugs), as opposed to a
# demand-side user-behavior story.
#
# Reuses the exact same agency list already pulled
# (hd_agencies_raw.csv) -- only the arrest data itself needs a
# fresh pull, since the offense code differs.
# ============================================================
import os, time, json, csv, threading
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

with open(os.path.join(BASE, '..', 'data', '.fbi_api_key.txt')) as f:
    KEY = f.read().strip()
API = "https://api.usa.gov/crime/fbi/cde"

OPIOID_COCAINE_SALE = "152"
SYNTH_NARCOTIC_SALE = "154"

OUT_ARRESTS = os.path.join(DATA, 'sale_county_arrests_raw.csv')
LOG = os.path.join(DATA, 'sale_pull_progress.log')
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

agencies = pd.read_csv(os.path.join(DATA, 'hd_agencies_raw.csv'))
log(f"Loaded {len(agencies)} agencies (reused from possession-arrest pull).")

done_oris = set()
if os.path.exists(OUT_ARRESTS):
    prev = pd.read_csv(OUT_ARRESTS)
    done_oris = set(prev['ori'].astype(str))
    log(f"Resuming: {len(done_oris)} already done.")
else:
    with open(OUT_ARRESTS, 'w', newline='', encoding='utf-8') as f:
        header = ['ori'] + [f'salecc_{y}' for y in range(2020, 2026)] + [f'salesyn_{y}' for y in range(2020, 2026)] + ['status']
        csv.writer(f).writerow(header)

todo = [r for _, r in agencies.iterrows() if str(r['ori']) not in done_oris]
log(f"To process: {len(todo)}")

def sum_year(series, year):
    suffix = f'-{year}'
    return sum(v for k, v in series.items() if k.endswith(suffix) and v is not None)

def process_one(row):
    ori = str(row['ori'])
    expected_key = f"{row['agency_name']} Arrests"
    cc_data = get_json(f"/arrest/agency/{ori}/{OPIOID_COCAINE_SALE}", "from=01-2020&to=12-2025&type=counts")
    syn_data = get_json(f"/arrest/agency/{ori}/{SYNTH_NARCOTIC_SALE}", "from=01-2020&to=12-2025&type=counts")
    if cc_data is None or syn_data is None:
        return [ori] + [''] * 12 + ['error']
    cc_act = (cc_data.get('actuals') or {}).get(expected_key)
    syn_act = (syn_data.get('actuals') or {}).get(expected_key)
    if cc_act is None or syn_act is None:
        return [ori] + [''] * 12 + ['no_exact_match']
    cc_vals = [sum_year(cc_act, y) for y in range(2020, 2026)]
    syn_vals = [sum_year(syn_act, y) for y in range(2020, 2026)]
    return [ori] + cc_vals + syn_vals + ['ok']

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
