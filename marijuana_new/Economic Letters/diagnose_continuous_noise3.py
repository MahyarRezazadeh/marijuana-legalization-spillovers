# Clean, unambiguous re-run: RADIUS = the actual distance cutoff applied to
# dist_2020_mi / dist_2023_mi directly (not an offset from the 50mi threshold).
# "50 miles beyond the switcher/control threshold" = radius 100 (50 + 50).
import os
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel_new.csv'))
main = panel[panel['d_od'].notna()].copy()

def cluster_fit(y, X, groups):
    Xc = sm.add_constant(X)
    g = pd.Categorical(groups).codes
    return sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})

print("RADIUS = actual cutoff applied to dist_2020_mi / dist_2023_mi (not an offset):\n")
for radius in [500, 300, 200, 150, 125, 100, 90, 75]:
    sub = main[(main['dist_2020_mi'] <= radius) | (main['dist_2023_mi'] <= radius)]
    n_sw = int((sub['switcher'] == 1).sum())
    n_ct = int((sub['switcher'] == 0).sum())
    r = cluster_fit(sub['d_od'], sub[['d_dist_mi']], sub['state_abbr'])
    print(f"  radius={radius:4d}mi  N={len(sub):4d} (sw={n_sw}, ct={n_ct}): "
          f"b={r.params['d_dist_mi']:+.5f} se={r.bse['d_dist_mi']:.5f} p={r.pvalues['d_dist_mi']:.4f}")

print("\n--- radius=100 (50mi threshold + 50mi buffer) in detail ---")
sub = main[(main['dist_2020_mi'] <= 100) | (main['dist_2023_mi'] <= 100)]
r = cluster_fit(sub['d_od'], sub[['d_dist_mi']], sub['state_abbr'])
print(f"N={len(sub)} (switchers={int((sub.switcher==1).sum())}, controls={int((sub.switcher==0).sum())})")
print(f"b={r.params['d_dist_mi']:.5f}  se={r.bse['d_dist_mi']:.5f}  p={r.pvalues['d_dist_mi']:.4f}")

sub.to_csv(os.path.join(DATA, 'overdose_panel_radius100.csv'), index=False)
import json
out = dict(radius=100, coef=r.params['d_dist_mi'], se=r.bse['d_dist_mi'], p=r.pvalues['d_dist_mi'],
           n=len(sub), n_switchers=int((sub.switcher==1).sum()), n_controls=int((sub.switcher==0).sum()))
with open(os.path.join(DATA, 'continuous_radius100_result.json'), 'w') as f:
    json.dump(out, f, indent=2)
