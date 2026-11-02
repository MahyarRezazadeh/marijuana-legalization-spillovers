# ============================================================
# Border-adjacency robustness check, part 2: stacked panel using
# LITERAL county-border adjacency to a legal-market state (from
# county_border_states.csv, built in build_border_adjacency.py) as
# the treatment definition, replacing the main design's continuous
# 50-mile distance-to-nearest-legal-county threshold. Same stacking
# logic as every other design in this project: drop always-near
# (already border-adjacent to a legal state as of the 2020
# baseline) counties, drop a county from later transitions once it
# switches, use each state's actual retail-sales date (not
# effective date) as the legal-market cutoff, with the same
# year-cutoff fix as elsewhere (2020 snapshot uses <=2019-12-31,
# all other years use <=year-12-31, since the panel's "2020"
# baseline reflects status at the START of 2020).
#
# A county with NO bordering state at all (fully interior) is a
# permanent control by construction (same role as a far/never-near
# county in the distance design) -- not an error or an exclusion.
# ============================================================
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
PARENT_DATA = os.path.join(BASE, '..', 'data')

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
print(f"Of those, counties with >=1 bordering state at all (legal or not): "
      f"{sample['fips5'].isin(county_borders.keys()).sum()}")

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

outcomes = pd.read_csv(os.path.join(PARENT_DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
od_cols = [f'od_{y}' for y in YEARS]
od = outcomes.drop_duplicates(subset='fips5')[['fips5'] + od_cols]
panel = panel.merge(od, on='fips5', how='left')
panel['od_pre'] = panel.apply(lambda r: r[f"od_{r['year_pre']}"], axis=1)
panel['od_post'] = panel.apply(lambda r: r[f"od_{r['year_post']}"], axis=1)
panel['od_rate_pre'] = panel['od_pre'] / panel['population'] * 1e5
panel['od_rate_post'] = panel['od_post'] / panel['population'] * 1e5
panel['d_od'] = panel['od_rate_post'] - panel['od_rate_pre']
panel['baseline_rate'] = panel['od_2020'] / panel['population'] * 1e5

n_before = panel.shape[0]
panel = panel.dropna(subset=['od_rate_pre', 'od_rate_post'])
print(f"Dropped {n_before - panel.shape[0]} rows with suppressed/missing overdose data; {panel.shape[0]} remain")

panel.to_csv(os.path.join(DATA, 'stacked_panel_border.csv'), index=False)
print(panel[['transition', 'switcher']].groupby('transition').agg(n=('switcher', 'size'), switches=('switcher', 'sum')))
