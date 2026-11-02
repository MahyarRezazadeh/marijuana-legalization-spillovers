# ============================================================
# Placebo check needed before claiming the 2nd-entrant speed effect is
# about CAPTIVE NONLEGAL demand specifically: is it actually just a
# generic "counties near ANY state line enter/compete faster" pattern
# (through-traffic, tourism, generic border-town agglomeration), which
# would have nothing to do with cannabis prohibition on the other side?
#
# Computes, for each of the 479 counties, distance to the nearest county
# in ANY different state (ignoring legal status), and re-estimates the
# rank-2 (1st->2nd entrant) Cox model with this generic-border distance
# in place of / alongside the nonlegal-specific distance measure.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from lifelines import CoxPHFitter

PROJ_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ_BASE, 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
EARTH_RADIUS_MI = 3958.8

panel = pd.read_csv(os.path.join(OUT, 'entry_hazard_panel_v2.csv'), dtype={'fips5': str})
centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)

own = panel[['fips5', 'state_abbr']].merge(centroids[['fips5', 'lat', 'lon']], on='fips5', how='left')

dist_any = np.full(len(own), np.nan)
for st in own['state_abbr'].unique():
    other = centroids[centroids['state_abbr'] != st]
    tree = BallTree(np.radians(other[['lat', 'lon']].to_numpy()), metric='haversine')
    mask = own['state_abbr'] == st
    d, _ = tree.query(np.radians(own.loc[mask, ['lat', 'lon']].to_numpy()), k=1)
    dist_any[mask.to_numpy()] = d[:, 0] * EARTH_RADIUS_MI

panel['dist_any_state_mi'] = dist_any
panel['log_dist_any_state'] = np.log(panel['dist_any_state_mi'])

print("Correlation, log(dist to nearest NONLEGAL county) vs log(dist to nearest ANY-state county):",
      panel[['log_dist_nonlegal', 'log_dist_any_state']].corr().iloc[0, 1].round(3))
print(f"Counties where nearest-any-state neighbor is itself non-legal (dist_any == dist_nonlegal): "
      f"{(np.isclose(panel['dist_any_state_mi'], panel['dist_nonlegal_mi'])).sum()} of {len(panel)}")

r2 = panel[panel['n_timed_entrants'] >= 1].dropna(
    subset=['second_gap_time', 'second_gap_event', 'log_dist_any_state', 'log_dist_nonlegal', 'log_population'])


def run_cox(df, covariates, label):
    d = df[['second_gap_time', 'second_gap_event', 'state_abbr'] + covariates].dropna()
    cph = CoxPHFitter()
    cph.fit(d, duration_col='second_gap_time', event_col='second_gap_event', strata=['state_abbr'])
    print(f"\n=== {label} (n={len(d)}, events={int(d['second_gap_event'].sum())}) ===")
    print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].round(4).to_string())
    return cph

run_cox(r2, ['log_dist_any_state', 'log_population'], 'PLACEBO: distance to nearest county in ANY other state')
run_cox(r2, ['log_dist_nonlegal', 'log_population'], 'Distance to nearest NONLEGAL county (main measure)')
run_cox(r2, ['log_dist_any_state', 'log_dist_nonlegal', 'log_population'], 'Both jointly')

# Sharpest version: restrict to counties whose nearest-any-state neighbor is a LEGAL
# state (so "nearest border" and "nearest nonlegal border" are two different places) --
# does proximity to that (legal, demand-irrelevant) neighbor still predict speed?
legal_neighbor = r2[~np.isclose(r2['dist_any_state_mi'], r2['dist_nonlegal_mi'])] if 'dist_any_state_mi' in r2.columns else None
legal_neighbor = panel.loc[r2.index]
legal_neighbor = legal_neighbor[~np.isclose(legal_neighbor['dist_any_state_mi'], legal_neighbor['dist_nonlegal_mi'])]
legal_neighbor = legal_neighbor.dropna(subset=['second_gap_time', 'second_gap_event', 'log_dist_any_state', 'log_population'])
print(f"\nSubsample where nearest-any-state neighbor is a LEGAL state (n={len(legal_neighbor)}):")
try:
    run_cox(legal_neighbor, ['log_dist_any_state', 'log_population'],
            'Counties whose NEAREST state neighbor is legal: does distance to THAT (legal) neighbor predict speed?')
except Exception as e:
    print(f"[skipped -- {e}]")

panel.to_csv(os.path.join(OUT, 'entry_hazard_panel_v3_with_placebo.csv'), index=False)
print("\nSaved entry_hazard_panel_v3_with_placebo.csv")
