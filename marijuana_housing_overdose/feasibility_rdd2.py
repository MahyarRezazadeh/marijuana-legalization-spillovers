import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

dist_df = pd.read_csv(os.path.join(DATA, 'rdd_border_distances.csv'), dtype={'fips5': str})

import json
with open(os.path.join(DATA, 'overdose_county_raw.json'), encoding='utf-8-sig') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od_raw['year'] = od_raw['year'].astype(int)

m = dist_df.merge(od_raw[['fips5', 'year', 'provisional_drug_overdose']], on=['fips5', 'year'], how='left')
m['usable'] = m['provisional_drug_overdose'].notna()

print("Per-PERIOD within-bandwidth non-suppressed counts (each year's own cross-section,")
print("as actually used by a pooled diff-in-disc regression -- no same-county-both-years restriction):\n")
for bw in [10, 25, 50, 75, 100]:
    for year in [2020, 2023]:
        sub = m[(m['year'] == year) & (m['dist_mi'] <= bw)]
        n_nl = len(sub[sub['side'] == 'never_legal'])
        n_nl_ok = sub[(sub['side'] == 'never_legal')]['usable'].sum()
        n_lg = len(sub[sub['side'] == 'legal'])
        n_lg_ok = sub[(sub['side'] == 'legal')]['usable'].sum()
        print(f"bw={bw:3d}mi  {year}  never-legal side: {n_nl_ok:4.0f}/{n_nl:4d} usable  |  "
              f"legal side: {n_lg_ok:4.0f}/{n_lg:4d} usable  |  total usable: {n_nl_ok+n_lg_ok:.0f}")
    print()
