# ============================================================
# New outcome, same design: business formation (Census Business
# Formation Statistics, county-level annual business applications,
# 2005-2025) as the outcome, using the EXACT same literal
# border-adjacency stacked design as build_stacked_border.py (which
# uses overdose deaths as the outcome). Same county universe, same
# 2020-2025 transitions, same always-near-at-2020-baseline drop,
# same switch-then-drop stacking rule, same retail_date legal-market
# cutoff.
#
# Outcome: change in business applications per 100k population
# (d_ba), county's own baseline (2020) BA rate as control -- mirrors
# d_od / baseline_rate exactly.
#
# Source: Census Bureau BFS county file (bfs_county_apps_annual.xlsx,
# downloaded fresh from census.gov/econ/bfs/data/county.html, no
# auth needed -- static bulk file, same pattern as every other
# Census source in this project). NOT the sector-detail version --
# BFS has no NAICS breakdown at the county level (only at national
# level), so this is the aggregate/all-sectors check only. Sector
# detail (the actual B1 ancillary-industry test) needs County
# Business Patterns instead, built separately.
# ============================================================
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
PARENT_DATA = os.path.join(BASE, '..', 'data')
SCRATCH = r'C:\Users\mahya\AppData\Local\Temp\claude\C--Users-mahya\ab92e63a-394d-4cbb-a614-fba6ce36efba\scratchpad'

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TRANSITIONS = list(zip(YEARS[:-1], YEARS[1:]))

legal = pd.read_csv(os.path.join(PARENT_DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
legal = legal.dropna(subset=['retail_date'])[['state_abbr', 'retail_date']]

def cutoff_date(year):
    return pd.Timestamp(f'{year - 1}-12-31') if year == 2020 else pd.Timestamp(f'{year}-12-31')

def legal_states_asof(year):
    d = cutoff_date(year)
    return set(legal.loc[legal['retail_date'] <= d, 'state_abbr'])

border = pd.read_csv(os.path.join(DATA, 'county_border_states.csv'), dtype={'fips5': str})
county_borders = border.groupby('fips5')['borders_state'].apply(set).to_dict()

dist = pd.read_csv(os.path.join(PARENT_DATA, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
universe = dist[['fips5', 'state_abbr', 'population']].copy()

def is_border_legal(fips5, legal_set):
    neighbors = county_borders.get(fips5, set())
    return len(neighbors & legal_set) > 0

legal_by_year = {y: legal_states_asof(y) for y in YEARS}
universe['border_legal_2020'] = universe['fips5'].apply(lambda f: is_border_legal(f, legal_by_year[2020]))

always_near = universe.loc[universe['border_legal_2020'], 'fips5']
sample = universe[~universe['fips5'].isin(always_near)].copy()
print(f"County universe: {len(universe)}; dropped {len(always_near)} always-border-adjacent-at-2020-baseline; {len(sample)} remain")

rows = []
switched = set()
for year_pre, year_post in TRANSITIONS:
    label = f'{year_pre}-{year_post}'
    active = sample[~sample['fips5'].isin(switched)]
    legal_pre, legal_post = legal_by_year[year_pre], legal_by_year[year_post]
    border_pre = active['fips5'].apply(lambda f: is_border_legal(f, legal_pre))
    border_post = active['fips5'].apply(lambda f: is_border_legal(f, legal_post))
    is_switcher = (~border_pre) & border_post
    chunk = pd.DataFrame({
        'fips5': active['fips5'],
        'state_abbr': active['state_abbr'],
        'population': active['population'],
        'transition': label,
        'year_pre': year_pre,
        'year_post': year_post,
        'switcher': is_switcher.astype(int).values,
    })
    rows.append(chunk)
    switched.update(active.loc[is_switcher, 'fips5'])

panel = pd.concat(rows, ignore_index=True)
print(f"\nStacked panel rows: {panel.shape[0]}, switch events: {panel['switcher'].sum()}")

# --- business-applications outcome ---
ba = pd.read_excel(os.path.join(SCRATCH, 'bfs_county_apps_annual.xlsx'), sheet_name='County Data', header=2)
ba['fips5'] = ba['County Code'].astype(int).astype(str).str.zfill(5)
ba_cols = {f'BA{y}': f'ba_{y}' for y in YEARS}
ba = ba.rename(columns=ba_cols)[['fips5'] + list(ba_cols.values())]
for c in ba_cols.values():
    ba[c] = pd.to_numeric(ba[c], errors='coerce')
n_nonnumeric = ba[list(ba_cols.values())].isna().sum().sum()
print(f"BFS county file: {ba.shape[0]} counties, years {YEARS}; non-numeric/unparseable cells: {n_nonnumeric}")

panel = panel.merge(ba, on='fips5', how='left')
n_before = panel.shape[0]
missing_ba = panel[[f'ba_{y}' for y in YEARS]].isna().any(axis=1).sum()
print(f"Rows missing BA data for at least one needed year: {missing_ba} of {n_before}")

panel['ba_pre'] = panel.apply(lambda r: r[f"ba_{r['year_pre']}"], axis=1)
panel['ba_post'] = panel.apply(lambda r: r[f"ba_{r['year_post']}"], axis=1)
panel['ba_rate_pre'] = panel['ba_pre'] / panel['population'] * 1e5
panel['ba_rate_post'] = panel['ba_post'] / panel['population'] * 1e5
panel['d_ba'] = panel['ba_rate_post'] - panel['ba_rate_pre']
panel['baseline_ba_rate'] = panel['ba_2020'] / panel['population'] * 1e5

panel = panel.dropna(subset=['ba_rate_pre', 'ba_rate_post', 'population'])
panel = panel[panel['population'] > 0]
print(f"Final analysis panel: {panel.shape[0]} rows, {panel['switcher'].sum()} switch events")

panel.to_csv(os.path.join(DATA, 'stacked_panel_border_bizform.csv'), index=False)
print(panel[['transition', 'switcher']].groupby('transition').agg(n=('switcher', 'size'), switches=('switcher', 'sum')))
print("\nSaved stacked_panel_border_bizform.csv")
