# ============================================================
# Mechanism check #4 (attention/awareness effect): state-level
# Google Trends search interest for "naloxone", tested around each
# state's own reference switch year (the modal switch year among
# its switcher counties, computed in state_reference_switch_year.csv).
#
# NOTE on scope: Google Trends' finest reliable geography is state
# (DMA-level exists but is sparse/unreliable for a systematic pull);
# this is necessarily a coarser, state-level, DESCRIPTIVE check, not
# a regression with clustered SEs like the other event studies in
# this project -- reported as suggestive, not a formally powered test.
#
# Cross-state comparability: Google Trends normalizes each query's
# own 0-100 scale independently, so raw levels are NOT comparable
# across states pulled in separate queries. Each state's series is
# therefore indexed to its own pre-period (event-time -3 to -1)
# average = 1.0 before pooling across states.
# ============================================================
import os, time, json
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

ref = pd.read_csv(os.path.join(DATA, 'state_reference_switch_year.csv'))
print(ref)

pytrends = TrendReq(hl='en-US', tz=360)
KEYWORD = 'naloxone'

rows = []
for _, r in ref.iterrows():
    state, ref_year = r['state_abbr'], int(r['ref_year'])
    start = f'{ref_year - 3}-01-01'
    end = f'{min(ref_year + 3, 2025)}-12-31'
    for attempt in range(4):
        try:
            pytrends.build_payload([KEYWORD], timeframe=f'{start} {end}', geo=f'US-{state}')
            df = pytrends.interest_over_time()
            break
        except Exception as e:
            print(f"  {state}: attempt {attempt+1} failed ({e}), retrying...")
            time.sleep(5 * (attempt + 1))
            df = None
    if df is None or df.empty:
        print(f"  {state}: NO DATA")
        continue
    df = df.reset_index()
    df['year'] = df['date'].dt.year
    annual = df.groupby('year')[KEYWORD].mean()
    for y, v in annual.items():
        rows.append({'state_abbr': state, 'ref_year': ref_year, 'year': y, 'search_interest': v})
    print(f"  {state}: ref_year={ref_year}, {len(annual)} years pulled")
    time.sleep(2)

trends = pd.DataFrame(rows)
trends.to_csv(os.path.join(DATA, 'google_trends_naloxone_raw.csv'), index=False)
print(f"\nSaved google_trends_naloxone_raw.csv: {len(trends)} state-year rows, {trends['state_abbr'].nunique()} states")
