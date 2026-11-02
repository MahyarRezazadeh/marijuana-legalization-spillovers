# ============================================================
# New outcome (2026-08-06): societal-attitude proxy via Google
# Trends search interest for "legalize marijuana" -- same
# structure as the existing naloxone mechanism check
# (pull_google_trends.py), reused because this project already
# established that Google Trends' finest RELIABLE geography is
# state (DMA-level was tried and found sparse/unreliable) --
# so this is a state-level descriptive check pegged to each
# state's reference switch year (state_reference_switch_year.csv,
# same 21 never-legal states), NOT the county-level 50-mile
# distance/switcher design used elsewhere in this project.
#
# Term choice: "legalize marijuana" (active/advocacy-leaning
# phrasing, closer to an attitude/support proxy) rather than the
# more neutral "marijuana legalization" (general news/informational
# interest) -- a judgment call, flagged for the user to revisit if
# a different phrasing is wanted.
#
# Cross-state comparability: Google Trends normalizes each query's
# own 0-100 scale independently per state pull, so raw levels are
# NOT comparable across states -- each state's series is indexed to
# its own pre-period (event-time -3 to -1) average = 1.0 before
# pooling, exactly as the naloxone check does.
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
KEYWORD = 'legalize marijuana'

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
trends.to_csv(os.path.join(DATA, 'google_trends_attitude_raw.csv'), index=False)
print(f"\nSaved google_trends_attitude_raw.csv: {len(trends)} state-year rows, {trends['state_abbr'].nunique()} states")
