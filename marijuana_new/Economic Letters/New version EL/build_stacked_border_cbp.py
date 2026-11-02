# ============================================================
# B1 sector-specific outcome: CBP establishment counts for five
# ancillary/non-plant-touching NAICS sectors, same literal
# border-adjacency stacked design as build_stacked_border.py /
# build_stacked_border_bizform.py. CBP's most recent available
# reference year is 2023, so this uses 3 transitions (2020-2021,
# 2021-2022, 2022-2023) instead of the usual 5.
#
# CBP's API omits zero-establishment county/sector/year cells
# entirely rather than returning explicit zeros -- after merging,
# missing = 0 establishments (a true zero, not suppressed/missing
# data; CBP does not suppress establishment counts, only payroll).
# ============================================================
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
PARENT_DATA = os.path.join(BASE, '..', 'data')

YEARS = [2020, 2021, 2022, 2023]
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

cbp = pd.read_csv(os.path.join(DATA, 'cbp_ancillary_raw.csv'), dtype={'fips5': str})
sectors = sorted(cbp['sector'].unique())
print(f"Sectors: {sectors}")

full = None
for sector in sectors:
    wide = cbp[cbp['sector'] == sector].pivot_table(
        index='fips5', columns='year', values='estab', aggfunc='sum').reset_index()
    wide.columns = ['fips5'] + [f'{sector}_{y}' for y in wide.columns[1:]]
    full = wide if full is None else full.merge(wide, on='fips5', how='outer')

# every county in the universe should appear even with all-zero cells
all_fips = pd.DataFrame({'fips5': universe['fips5'].unique()})
full = all_fips.merge(full, on='fips5', how='left')
val_cols = [c for c in full.columns if c != 'fips5']
full[val_cols] = full[val_cols].fillna(0)

panel = panel.merge(full, on='fips5', how='left')

for sector in sectors:
    panel[f'{sector}_pre'] = panel.apply(lambda r, s=sector: r[f"{s}_{r['year_pre']}"], axis=1)
    panel[f'{sector}_post'] = panel.apply(lambda r, s=sector: r[f"{s}_{r['year_post']}"], axis=1)
    panel[f'{sector}_rate_pre'] = panel[f'{sector}_pre'] / panel['population'] * 1e5
    panel[f'{sector}_rate_post'] = panel[f'{sector}_post'] / panel['population'] * 1e5
    panel[f'd_{sector}'] = panel[f'{sector}_rate_post'] - panel[f'{sector}_rate_pre']
    panel[f'baseline_{sector}_rate'] = panel[f'{sector}_2020'] / panel['population'] * 1e5

panel = panel[panel['population'] > 0].copy()
panel.to_csv(os.path.join(DATA, 'stacked_panel_border_cbp.csv'), index=False)
print(f"\nFinal panel: {panel.shape[0]} rows, {panel['switcher'].sum()} switch events")
print(panel[['transition', 'switcher']].groupby('transition').agg(n=('switcher', 'size'), switches=('switcher', 'sum')))
print("\nSaved stacked_panel_border_cbp.csv")
