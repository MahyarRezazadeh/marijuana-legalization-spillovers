# ============================================================
# Diagnoses why the continuous per-mile distance specification
# is so much noisier / less informative than the binary switcher
# spec, despite using "more" of the underlying variation.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel_new.csv'))
main = panel[panel['d_od'].notna()].copy()
print(f"N = {len(main)}\n")

def cluster_fit(y, X, groups):
    Xc = sm.add_constant(X)
    g = pd.Categorical(groups).codes
    return sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})

# ── 1. Distribution of the running variable ──
print("=== 1. Distribution of d_dist_mi (896 non-suppressed counties) ===")
print(main['d_dist_mi'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
print(f"\nSwitchers: dist_2020={main[main.switcher==1]['dist_2020_mi'].describe()[['min','25%','50%','75%','max']].to_dict()}")

# ── 2. Leverage / influence diagnostics on the baseline OLS ──
print("\n=== 2. Influence diagnostics (Cook's distance) ===")
res_full = cluster_fit(main['d_od'], main[['d_dist_mi']], main['state_abbr'])
ols_plain = sm.OLS(main['d_od'], sm.add_constant(main[['d_dist_mi']])).fit()
infl = ols_plain.get_influence()
cooks_d = infl.cooks_distance[0]
main['cooks_d'] = cooks_d
top10 = main.nlargest(10, 'cooks_d')[['fips5', 'state_abbr', 'switcher', 'd_dist_mi', 'd_od', 'cooks_d']]
print(top10.to_string(index=False))
print(f"\nBaseline (all N={len(main)}): b={res_full.params['d_dist_mi']:.5f} p={res_full.pvalues['d_dist_mi']:.3f}")

# Drop top 1% / 5% most influential
for pct in [0.01, 0.05]:
    cutoff = main['cooks_d'].quantile(1 - pct)
    trimmed = main[main['cooks_d'] < cutoff]
    r = cluster_fit(trimmed['d_od'], trimmed[['d_dist_mi']], trimmed['state_abbr'])
    print(f"Excluding top {pct:.0%} Cook's D (N={len(trimmed)}): "
          f"b={r.params['d_dist_mi']:.5f} se={r.bse['d_dist_mi']:.5f} p={r.pvalues['d_dist_mi']:.3f}")

# ── 3. Winsorizing extreme distance changes ──
print("\n=== 3. Winsorizing d_dist_mi ===")
for pct in [0.01, 0.05, 0.10]:
    lo, hi = main['d_dist_mi'].quantile([pct, 1-pct])
    w = main.copy()
    w['d_dist_mi_w'] = w['d_dist_mi'].clip(lo, hi)
    r = cluster_fit(w['d_od'], w[['d_dist_mi_w']], w['state_abbr'])
    print(f"Winsorized at {pct:.0%}/{1-pct:.0%}: b={r.params['d_dist_mi_w']:.5f} "
          f"se={r.bse['d_dist_mi_w']:.5f} p={r.pvalues['d_dist_mi_w']:.3f}")

# ── 4. Is the effect local? Restrict to counties "near the margin" ──
print("\n=== 4. Restricting to counties near the 50mi margin (local effect test) ===")
for bw in [500, 300, 200, 150, 100]:
    # near the margin = started OR ended within bw miles of the 50mi cutoff
    near_margin = main[(main['dist_2020_mi'] <= 50 + bw) | (main['dist_2023_mi'] <= 50 + bw)]
    if len(near_margin) < 20:
        continue
    r = cluster_fit(near_margin['d_od'], near_margin[['d_dist_mi']], near_margin['state_abbr'])
    n_sw = int((near_margin['switcher']==1).sum())
    print(f"Within {bw}mi of margin (N={len(near_margin)}, switchers={n_sw}): "
          f"b={r.params['d_dist_mi']:.5f} se={r.bse['d_dist_mi']:.5f} p={r.pvalues['d_dist_mi']:.3f}")

# ── 5. Suppression asymmetry by distance ──
print("\n=== 5. Does suppression correlate with distance-change magnitude? ===")
full_all = pd.read_csv(os.path.join(DATA, 'overdose_county_panel_new.csv'))
full_all['suppressed'] = full_all['d_od'].isna()
print(full_all.groupby('suppressed')['dist_change_mi'].describe()[['mean','50%','std']])
print(f"\nSwitcher suppression rate: {full_all[full_all.switcher==1]['suppressed'].mean():.1%} "
      f"(N={len(full_all[full_all.switcher==1])})")
print(f"Control suppression rate: {full_all[full_all.sample_grp=='Control']['suppressed'].mean():.1%} "
      f"(N={len(full_all[full_all.sample_grp=='Control'])})")

# ── 6. Variance decomposition: how much of the switcher effect's identifying
#      power comes from far-away "control" counties whose distance barely moved? ──
print("\n=== 6. Correlation between switcher status and d_dist_mi ===")
print(main[['switcher', 'd_dist_mi']].corr())
print(f"\nMean d_dist_mi: switchers={main[main.switcher==1]['d_dist_mi'].mean():.1f}, "
      f"controls={main[main.switcher==0]['d_dist_mi'].mean():.1f}")
