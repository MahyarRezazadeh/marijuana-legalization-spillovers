# ============================================================
# Rebuild of build_switch_panel.py using the CORRECTED revenue/dispensary
# panel (revenue_perstore_border70_panel_CORRECTED.csv, n=787), which fixes
# Washington's undercounted dispensary registry (98 -> 480 active licenses
# statewide). Colorado rows are numerically identical to the original panel
# (Colorado's registry was never affected), so the CO-NM switch panel is
# rebuilt only for verification; the WA-OR switch panel is expected to change
# because WA's dispensary_count -- and therefore revenue_per_store -- changes.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
EARTH_RADIUS_MI = 3958.8
THRESH_MI = 70

centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)


def min_dist_to_state(from_df, to_state_abbr):
    target = centroids[centroids['state_abbr'] == to_state_abbr]
    tree = BallTree(np.radians(target[['lat', 'lon']].to_numpy()), metric='haversine')
    dist_rad, _ = tree.query(np.radians(from_df[['lat', 'lon']].to_numpy()), k=1)
    return dist_rad[:, 0] * EARTH_RADIUS_MI


panel = pd.read_csv(os.path.join(BASE, 'tables', 'revenue_perstore_border70_panel_CORRECTED.csv'),
                     dtype={'fips5': str})

# ---------- Colorado x New Mexico (verification rebuild) ----------
co = panel[panel['state_abbr'] == 'CO'].copy()
co['dist_nm_mi'] = min_dist_to_state(co, 'NM')
co['nm_border_70mi'] = (co['dist_nm_mi'] <= THRESH_MI).astype(int)
co['other_border_70mi'] = ((co['border_70mi'] == 1) & (co['nm_border_70mi'] == 0)).astype(int)
co['interior'] = ((co['nm_border_70mi'] == 0) & (co['other_border_70mi'] == 0)).astype(int)
co['event_year'] = co['fy_year'] - 2022
co.to_csv(os.path.join(OUT, 'co_nm_switch_panel_CORRECTED.csv'), index=False)
print("CO-NM panel (corrected):", co.shape,
      "| NM-border counties:", co.loc[co['nm_border_70mi'] == 1, 'fips5'].nunique(),
      "| other-border counties:", co.loc[co['other_border_70mi'] == 1, 'fips5'].nunique(),
      "| interior counties:", co.loc[co['interior'] == 1, 'fips5'].nunique())
print(co.loc[co['nm_border_70mi'] == 1, 'county_name'].unique())

# ---------- Washington x Oregon / Idaho (the one that should change) ----------
wa = panel[panel['state_abbr'] == 'WA'].copy()
wa['dist_or_mi'] = min_dist_to_state(wa, 'OR')
wa['dist_id_mi'] = min_dist_to_state(wa, 'ID')
wa['or_border_70mi'] = (wa['dist_or_mi'] <= THRESH_MI).astype(int)
wa['id_border_70mi'] = ((wa['dist_id_mi'] <= THRESH_MI) & (wa['or_border_70mi'] == 0)).astype(int)
wa['interior'] = ((wa['or_border_70mi'] == 0) & (wa['id_border_70mi'] == 0)).astype(int)
wa['event_year'] = wa['fy_year'] - 2016
wa['post_clean'] = (wa['fy_year'] >= 2017).astype(int)
wa['post_any'] = (wa['fy_year'] >= 2016).astype(int)
wa.to_csv(os.path.join(OUT, 'wa_or_switch_panel_CORRECTED.csv'), index=False)
print("\nWA-OR panel (corrected):", wa.shape,
      "| OR-border counties:", wa.loc[wa['or_border_70mi'] == 1, 'fips5'].nunique(),
      "| ID-border(control) counties:", wa.loc[wa['id_border_70mi'] == 1, 'fips5'].nunique(),
      "| interior counties:", wa.loc[wa['interior'] == 1, 'fips5'].nunique())
print(wa.loc[wa['or_border_70mi'] == 1, 'county_name'].unique())
print(wa.loc[wa['id_border_70mi'] == 1, 'county_name'].unique())
