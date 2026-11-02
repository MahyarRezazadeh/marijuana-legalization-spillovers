# ============================================================
# Replaces the two-snapshot (Dec 2020 / Dec 2023) design with a
# stacked multi-period panel using EVERY state's actual
# legalization date, for both years CDC VSRR county-level data
# is available (Dec 2020 through Dec 2025, 6 annual snapshots,
# 5 year-over-year transitions).
#
# For each of the 5 transitions (2020->21, 21->22, 22->23, 23->24,
# 24->25), a county is a "new switcher" if its distance to the
# nearest legal-market county crossed from >=50mi to <50mi exactly
# in that transition. Once a county switches, it is dropped from
# all LATER transitions (it is no longer "at risk" of switching,
# and keeping it in as an always-treated unit would reintroduce
# the classic already-treated-as-control TWFE bias). Counties that
# are already <50mi as of the Dec-2020 baseline are "always near"
# and dropped from every transition (same as the original design).
#
# Built for TWO treatment definitions, run separately:
#   - recreational: legal-market county = retail-sales-open county
#     (legal_dates_new.csv), the paper's primary design
#   - medical: legal-market county = medical-access-open county
#     (medical_dates.csv), a robustness check
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MARIJUANA_NEW = os.path.dirname(BASE)
ROOT = os.path.dirname(MARIJUANA_NEW)
DATA = os.path.join(BASE, 'data')
HOUSING_DATA = os.path.join(ROOT, 'marijuana_housing_overdose', 'data')

THRESH = 50
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
SNAPSHOTS = {y: pd.Timestamp(f'{y}-12-31') for y in YEARS}

gaz = pd.read_csv(os.path.join(HOUSING_DATA, 'gaz', '2023_Gaz_counties_national.txt'), sep='\t')
gaz.columns = [c.strip() for c in gaz.columns]
gaz['GEOID'] = gaz['GEOID'].astype(str).str.zfill(5)
gaz['USPS'] = gaz['USPS'].str.strip()

panel = pd.read_stata(os.path.join(MARIJUANA_NEW, 'data', 'analysis_panel.dta'))
panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
base_meta = (panel[panel['year'] == 2020]
             [['fips5', 'state_abbr', 'population', 'lat', 'lon']]
             .drop_duplicates('fips5'))

def haversine_min(lat0, lon0, lats, lons):
    R = 3958.8
    p0, p1 = np.radians(lat0), np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dlmb / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))
    return d.min()

def compute_year_distances(meta, legal_state_set):
    legal_gaz = gaz[gaz['USPS'].isin(legal_state_set)]
    llat, llon = legal_gaz['INTPTLAT'].to_numpy(), legal_gaz['INTPTLONG'].to_numpy()
    dists = np.empty(len(meta))
    for i, (_, row) in enumerate(meta.iterrows()):
        dists[i] = haversine_min(row['lat'], row['lon'], llat, llon)
    return dists

def build_design(label, legal_dates_csv, date_col, universe_states=None):
    print(f"\n=== Building {label} design ===")
    legal = pd.read_csv(os.path.join(DATA, legal_dates_csv), parse_dates=[date_col])
    legal = legal.dropna(subset=[date_col])

    if universe_states is None:
        meta = base_meta[base_meta['state_abbr'] != 'ME'].copy()
    else:
        meta = base_meta[base_meta['state_abbr'].isin(universe_states)].copy()
    meta = meta.reset_index(drop=True)
    print(f"County universe: {meta['state_abbr'].nunique()} states, {len(meta)} counties")

    dist_by_year = {}
    for y in YEARS:
        legal_states_y = set(legal[legal[date_col] <= SNAPSHOTS[y]]['state_abbr'])
        dist_by_year[y] = compute_year_distances(meta, legal_states_y)
        print(f"  {y}: {len(legal_states_y)} legal-market states -> distances computed")

    for y in YEARS:
        meta[f'dist_{y}'] = dist_by_year[y]

    always_near = meta['dist_2020'] < THRESH
    print(f"Always-near (dropped): {always_near.sum()} counties")
    meta = meta[~always_near].copy()

    rows = []
    switched_already = pd.Series(False, index=meta.index)
    for t in YEARS[1:]:
        t0 = t - 1
        still_at_risk = ~switched_already & (meta[f'dist_{t0}'] >= THRESH)
        sub = meta[still_at_risk].copy()
        new_switch = sub[f'dist_{t}'] < THRESH
        sub['switcher'] = new_switch.astype(int)
        sub['d_dist_mi'] = sub[f'dist_{t}'] - sub[f'dist_{t0}']
        sub['transition'] = f'{t0}-{t}'
        sub['year_pre'] = t0
        sub['year_post'] = t
        rows.append(sub[['fips5', 'state_abbr', 'population', 'transition', 'year_pre', 'year_post',
                          'switcher', 'd_dist_mi']])
        switched_already.loc[sub.index[new_switch.values]] = True

    stacked = pd.concat(rows, ignore_index=True)
    n_sw = int(stacked['switcher'].sum())
    print(f"Stacked panel: {len(stacked)} county-transition rows, {n_sw} switch events "
          f"across {len(YEARS)-1} transitions")
    stacked.to_csv(os.path.join(DATA, f'stacked_panel_{label}.csv'), index=False)
    meta.to_csv(os.path.join(DATA, f'distance_by_year_{label}.csv'), index=False)
    return stacked

# ── Recreational (primary design): same 31-state universe as before ──
build_design('rec', 'legal_dates_new.csv', 'retail_date')

# ── Medical (robustness): drop states already medically legal by the
#    Dec-2020 baseline (their own counties are trivially "near" from
#    the start and contribute no switcher/control variation) ──
med = pd.read_csv(os.path.join(DATA, 'medical_dates.csv'), parse_dates=['medical_access_date'])
already_legal_states = set(med[med['medical_access_date'] <= SNAPSHOTS[2020]]['state_abbr'])
full_universe = set(base_meta['state_abbr'].unique()) - {'ME'}
med_universe = sorted(full_universe - already_legal_states)
print(f"\nMedical-design universe: {len(med_universe)} states "
      f"(excluded {len(already_legal_states)} already medically-legal by 2020: {sorted(already_legal_states)})")
build_design('med', 'medical_dates.csv', 'medical_access_date', universe_states=med_universe)

print("\nDone.")
