# ============================================================
# Rebuilds the switcher/control distance-to-border treatment
# panel using the user-supplied authoritative recreational-
# marijuana timing data (effective date AND first retail sales
# date per state). Uses FIRST RETAIL SALES as the operative
# "legal-market" cutoff (not bare policy-effective date), since
# the whole design is about proximity to a place people can
# actually buy from -- this also cleanly handles Virginia (never
# opened retail in-sample) and pushes Ohio/Delaware's retail
# past the Dec-2023 snapshot without any special-casing.
# Same treatment design as before (50-mile switcher/control
# radius, Haversine county-centroid distance, Dec-2020 vs
# Dec-2023 snapshots) -- only the underlying legal-state-timing
# data changes.
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

# ── New authoritative legalization timing ──
legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['effective_date', 'retail_date'])
SNAPSHOT_2020 = pd.Timestamp('2020-12-31')
SNAPSHOT_2023 = pd.Timestamp('2023-12-31')

legal_states_2020 = set(legal[legal['retail_date'] <= SNAPSHOT_2020]['state_abbr'])
legal_states_2023 = set(legal[legal['retail_date'] <= SNAPSHOT_2023]['state_abbr'])
never_open_by_2023 = set(legal['state_abbr']) - legal_states_2023  # VA (no retail), DE, OH (retail after window)

print(f"Legal-market (retail open) as of Dec 2020: {sorted(legal_states_2020)}")
print(f"Legal-market (retail open) as of Dec 2023: {sorted(legal_states_2023)}")
print(f"In the new dataset but NOT yet retail-open by Dec 2023: {sorted(never_open_by_2023)}")

# ── Gazetteer county centroids (reuse the already-downloaded file) ──
gaz = pd.read_csv(os.path.join(HOUSING_DATA, 'gaz', '2023_Gaz_counties_national.txt'), sep='\t')
gaz.columns = [c.strip() for c in gaz.columns]
gaz['GEOID'] = gaz['GEOID'].astype(str).str.zfill(5)
gaz['USPS'] = gaz['USPS'].str.strip()

# ── Companion panel: county universe, population, state_abbr, lat/lon ──
# (32 never-legal states as originally defined by the companion project;
# Maine now needs to be REMOVED from this set since its retail market
# opened Oct 2020, before even our 2020 baseline snapshot.)
panel = pd.read_stata(os.path.join(MARIJUANA_NEW, 'data', 'analysis_panel.dta'))
panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
meta = (panel[panel['year'] == 2020]
        [['fips5', 'state_abbr', 'population', 'lat', 'lon']]
        .drop_duplicates('fips5'))
n_before = meta['state_abbr'].nunique()
meta = meta[meta['state_abbr'] != 'ME'].copy()
print(f"\nCompanion never-legal-state panel: {n_before} states -> {meta['state_abbr'].nunique()} "
      f"after removing Maine (now retail-legal since Oct 2020). {len(meta)} counties remain.")

def haversine_min(lat0, lon0, lats, lons):
    R = 3958.8
    p0, p1 = np.radians(lat0), np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi/2)**2 + np.cos(p0)*np.cos(p1)*np.sin(dlmb/2)**2
    d = 2*R*np.arcsin(np.sqrt(a))
    return d.min()

def compute_distances(legal_state_set, label):
    legal_gaz = gaz[gaz['USPS'].isin(legal_state_set)]
    llat, llon = legal_gaz['INTPTLAT'].to_numpy(), legal_gaz['INTPTLONG'].to_numpy()
    dists = []
    for _, row in meta.iterrows():
        d = haversine_min(row['lat'], row['lon'], llat, llon)
        dists.append(d)
    print(f"  {label}: done ({len(dists)} counties)")
    return np.array(dists)

meta['dist_2020_mi'] = compute_distances(legal_states_2020, 'Dec 2020 distances')
meta['dist_2023_mi'] = compute_distances(legal_states_2023, 'Dec 2023 distances')
meta['dist_change_mi'] = meta['dist_2023_mi'] - meta['dist_2020_mi']

always_near = (meta['dist_2020_mi'] < THRESH) & (meta['dist_2023_mi'] < THRESH)
switcher = (meta['dist_2020_mi'] >= THRESH) & (meta['dist_2023_mi'] < THRESH)
meta['switcher'] = switcher.astype(int)
meta['sample_grp'] = np.select([always_near, switcher], ['Always near rec border', 'Switcher'], default='Control')

print(f"\nNew treatment panel: {len(meta)} counties. "
      f"always_near={always_near.sum()}, switcher={switcher.sum()}, control={(~always_near & ~switcher).sum()}")

meta.to_csv(os.path.join(DATA, 'treatment_panel_new.csv'), index=False)
print(f"Saved treatment_panel_new.csv")
