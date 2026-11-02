"""Quick feasibility check: merge NASS land values onto the existing distance/county panel,
check coverage, and look at a naive raw pattern (no causal claim -- just checking the merge works
and whether there's anything worth building further)."""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, '..', 'Economic Letters', 'data')

dist = pd.read_csv(os.path.join(SRC, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})

nass = pd.read_csv(os.path.join(BASE, 'data', 'nass_land_value_per_acre_county.csv'), dtype={'STATE_FIPS_CODE': str, 'COUNTY_ANSI': str})
nass = nass[nass['VALUE'] != '(D)'].copy()
nass['VALUE'] = nass['VALUE'].str.replace(',', '').astype(float)
nass['YEAR'] = nass['YEAR'].astype(int)
nass = nass.dropna(subset=['STATE_FIPS_CODE', 'COUNTY_ANSI'])
nass['fips5'] = nass['STATE_FIPS_CODE'].str.zfill(2) + nass['COUNTY_ANSI'].str.zfill(3)

wide = nass.pivot_table(index='fips5', columns='YEAR', values='VALUE', aggfunc='first')
wide.columns = [f'landval_{c}' for c in wide.columns]
wide = wide.reset_index()

merged = dist.merge(wide, on='fips5', how='inner')
print(f"Distance-panel counties: {len(dist)}")
print(f"NASS land-value counties (any year): {wide['fips5'].nunique()}")
print(f"Matched (inner join): {len(merged)}")
print(f"Matched with all 3 years present: {merged.dropna(subset=['landval_2012','landval_2017','landval_2022']).shape[0]}")

m = merged.dropna(subset=['landval_2012', 'landval_2022']).copy()
m['pct_change_2012_2022'] = (m['landval_2022'] - m['landval_2012']) / m['landval_2012']

m['switch_by_2022'] = (m['dist_2020'] >= 50) & (m['dist_2025'] < 50)
m['always_far'] = (m['dist_2020'] >= 50) & (m['dist_2025'] >= 50)

print("\nRaw land-value growth 2012->2022, switcher vs always-far (NO controls, purely descriptive):")
print(m.groupby(m['switch_by_2022'].map({True: 'became_near_by_2025', False: 'other'}))['pct_change_2012_2022'].agg(['mean', 'median', 'count']))
print("\nalways-far subset only:")
print(m[m['always_far']]['pct_change_2012_2022'].agg(['mean', 'median', 'count']))

m.to_csv(os.path.join(BASE, 'data', 'land_value_preview_panel.csv'), index=False)
print("\nSaved land_value_preview_panel.csv")
