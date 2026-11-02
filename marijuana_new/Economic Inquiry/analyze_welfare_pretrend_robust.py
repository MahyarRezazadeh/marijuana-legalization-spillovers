# ============================================================
# Welfare translation using the continuous per-mile coefficient
# (which survives the pre-trends/state-FE battery) applied to
# REALIZED distance changes across the full at-risk sample --
# not just the binary switcher indicator, which does not survive
# state fixed effects. This is the pre-trends-robust companion to
# the original switcher-based welfare number.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0) & (s['d_dist_mi'] < 0)].copy()  # counties that got closer
s['g'] = pd.Categorical(s['state_abbr']).codes

# baseline-level-adjusted continuous coefficient (from reestimate_main_with_controls.py)
main_ctrl = pd.read_csv(os.path.join(DATA, 'main_with_controls.csv'))
coef_row = main_ctrl[(main_ctrl['x'] == 'd_dist_mi') & (main_ctrl['spec'] == '+ baseline overdose level')].iloc[0]
b, se = coef_row['coef'], coef_row['se']
print(f"Baseline-level-adjusted continuous coefficient: b={b:.5f} se={se:.5f}")

# implied averted deaths = -b * (sum over counties that got closer of |d_dist_mi| * population/1e5)
s['implied_effect'] = -b * s['d_dist_mi'] * s['population'] / 1e5  # d_dist_mi negative -> positive effect
total_averted = s['implied_effect'].sum()
total_averted_lo = -(b + 1.96*se) * (s['d_dist_mi'] * s['population'] / 1e5).sum()
total_averted_hi = -(b - 1.96*se) * (s['d_dist_mi'] * s['population'] / 1e5).sum()

print(f"Counties that got closer to a legal-market border: {s['fips5'].nunique()}")
print(f"Implied averted deaths (continuous, realized distance changes): {total_averted:.1f} "
      f"[{min(total_averted_lo,total_averted_hi):.1f}, {max(total_averted_lo,total_averted_hi):.1f}]")

with open(os.path.join(DATA, 'welfare_summary_ei.json')) as f:
    old = json.load(f)
le_diff = old['le_diff']
yll = total_averted * le_diff
print(f"Years of life saved (using LE gap = {le_diff:.2f}): {yll:,.0f}")

out = dict(coef=b, se=se, n_counties=int(s['fips5'].nunique()),
           averted_deaths=total_averted, averted_lo=min(total_averted_lo,total_averted_hi),
           averted_hi=max(total_averted_lo,total_averted_hi), le_diff=le_diff, yll=yll)
with open(os.path.join(DATA, 'welfare_summary_pretrend_robust.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved welfare_summary_pretrend_robust.json")
