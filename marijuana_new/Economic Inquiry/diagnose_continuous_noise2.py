import os
import numpy as np
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

print("Fine-grained bandwidth sweep:")
for bw in [500, 400, 350, 300, 275, 250, 225, 200, 175, 150, 125, 100, 75]:
    near_margin = main[(main['dist_2020_mi'] <= 50 + bw) | (main['dist_2023_mi'] <= 50 + bw)]
    if len(near_margin) < 20:
        continue
    r = cluster_fit(near_margin['d_od'], near_margin[['d_dist_mi']], near_margin['state_abbr'])
    n_sw = int((near_margin['switcher'] == 1).sum())
    n_ct = int((near_margin['switcher'] == 0).sum())
    print(f"  bw={bw:4d}mi  N={len(near_margin):4d} (sw={n_sw}, ct={n_ct}): "
          f"b={r.params['d_dist_mi']:+.5f} se={r.bse['d_dist_mi']:.5f} p={r.pvalues['d_dist_mi']:.3f}")

# Check for outlier-driven results specifically WITHIN the bw=150 sample
print("\nOutlier check within the bw=150 sample:")
near150 = main[(main['dist_2020_mi'] <= 200) | (main['dist_2023_mi'] <= 200)].copy()
ols_plain = sm.OLS(near150['d_od'], sm.add_constant(near150[['d_dist_mi']])).fit()
infl = ols_plain.get_influence()
near150['cooks_d'] = infl.cooks_distance[0]
print(near150.nlargest(5, 'cooks_d')[['fips5','state_abbr','switcher','dist_2020_mi','dist_2023_mi','d_dist_mi','d_od','cooks_d']].to_string(index=False))
trimmed = near150[near150['cooks_d'] < near150['cooks_d'].quantile(0.95)]
r = cluster_fit(trimmed['d_od'], trimmed[['d_dist_mi']], trimmed['state_abbr'])
print(f"bw=150, excluding top 5% Cook's D (N={len(trimmed)}): b={r.params['d_dist_mi']:.5f} "
      f"se={r.bse['d_dist_mi']:.5f} p={r.pvalues['d_dist_mi']:.3f}")

# Compare: what if we use log-distance in the SAME restricted sample?
r_log = cluster_fit(near150['d_od'], near150[['d_dist_mi']], near150['state_abbr'])
print(f"\nFor reference, bw=150 unrestricted-outlier version: b={r_log.params['d_dist_mi']:.5f} p={r_log.pvalues['d_dist_mi']:.3f}")
