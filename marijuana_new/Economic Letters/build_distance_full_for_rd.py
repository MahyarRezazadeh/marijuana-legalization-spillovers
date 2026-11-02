# ============================================================
# Rebuilds county distances for the FULL 2,331-county universe
# (31 never-rec-legal-at-baseline states), WITHOUT dropping the
# 116 "always-near" counties -- needed for the diff-in-disc
# appendix check, which requires observations on both sides of
# the 50-mile cutoff in every year. The main design's saved
# distance file (distance_by_year_rec.csv) excludes these
# counties since they contribute no switcher/control variation to
# the stacked design; the RD design needs them back.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MARIJUANA_NEW = os.path.dirname(BASE)
ROOT = os.path.dirname(MARIJUANA_NEW)
DATA = os.path.join(BASE, 'data')
HOUSING_DATA = os.path.join(ROOT, 'marijuana_housing_overdose', 'data')

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
SNAPSHOTS = {y: pd.Timestamp(f'{y}-12-31') for y in YEARS}

gaz = pd.read_csv(os.path.join(HOUSING_DATA, 'gaz', '2023_Gaz_counties_national.txt'), sep='\t')
gaz.columns = [c.strip() for c in gaz.columns]
gaz['GEOID'] = gaz['GEOID'].astype(str).str.zfill(5)
gaz['USPS'] = gaz['USPS'].str.strip()

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date']).dropna(subset=['retail_date'])

panel = pd.read_stata(os.path.join(MARIJUANA_NEW, 'data', 'analysis_panel.dta'))
panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
meta = (panel[panel['year'] == 2020][['fips5', 'state_abbr', 'population', 'lat', 'lon']]
        .drop_duplicates('fips5'))
meta = meta[meta['state_abbr'] != 'ME'].copy().reset_index(drop=True)
print(f"Full county universe (no always-near drop): {len(meta)} counties")

def haversine_min(lat0, lon0, lats, lons):
    R = 3958.8
    p0, p1 = np.radians(lat0), np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dlmb / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))
    return d.min()

def compute_year_distances(legal_state_set):
    legal_gaz = gaz[gaz['USPS'].isin(legal_state_set)]
    llat, llon = legal_gaz['INTPTLAT'].to_numpy(), legal_gaz['INTPTLONG'].to_numpy()
    dists = np.empty(len(meta))
    for i, (_, row) in enumerate(meta.iterrows()):
        dists[i] = haversine_min(row['lat'], row['lon'], llat, llon)
    return dists

for y in YEARS:
    legal_states_y = set(legal[legal['retail_date'] <= SNAPSHOTS[y]]['state_abbr'])
    meta[f'dist_{y}'] = compute_year_distances(legal_states_y)
    n_near = int((meta[f'dist_{y}'] < 50).sum())
    print(f"  {y}: {len(legal_states_y)} legal states, {n_near} counties <50mi")

meta.to_csv(os.path.join(DATA, 'distance_by_year_rec_full.csv'), index=False)
print("Saved distance_by_year_rec_full.csv")
