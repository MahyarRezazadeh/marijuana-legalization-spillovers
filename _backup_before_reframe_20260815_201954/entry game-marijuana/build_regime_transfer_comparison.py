# ============================================================
# Regime-level fiscal transfer comparison: how much 2025 border-county
# revenue is attributable to the border effect (i.e. "transferred"
# from cross-border demand in neighboring non-legal states), pooled
# for the capped regime (CO+WA, using the pooled, significant Table 1
# coefficient) vs Montana (moratorium regime, using its own
# significant revenue-per-capita coefficient). Reuses only numbers
# already estimated this session (fiscal_spillover_by_state.csv,
# revenue_perstore_border70_panel_CORRECTED.csv, montana_revenue_
# perstore_cross.csv) -- no new regressions.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

spill = pd.read_csv(os.path.join(TABLES, 'fiscal_spillover_by_state.csv')).set_index('state')

# ---------- Capped regime: CO+WA pooled, using the pooled Table 1 coefficient ----------
pooled_b, pooled_p = 0.341, 0.033  # Table 1, full-control binary spec (already in manuscript)
pooled_premium = np.exp(pooled_b) - 1

cowa_border_rev = spill.loc[['CO', 'WA'], 'border_rev_2025'].sum()
cowa_state_total = spill.loc[['CO', 'WA'], 'state_total_2025'].sum()
cowa_attributable = cowa_border_rev * (pooled_premium / (1 + pooled_premium))

# ---------- Moratorium regime: Montana, using its own significant per-capita coefficient ----------
mt_b, mt_p = spill.loc['MT', 'coef'], spill.loc['MT', 'p']
mt_premium = np.exp(mt_b) - 1
mt_border_rev = spill.loc['MT', 'border_rev_2025']
mt_state_total = spill.loc['MT', 'state_total_2025']
mt_attributable = spill.loc['MT', 'attributable_usd']

print("=" * 70)
print("REGIME-LEVEL FISCAL TRANSFER COMPARISON")
print("=" * 70)
print(f"\nCapped regime (CO+WA pooled, b={pooled_b}, p={pooled_p}, premium={pooled_premium*100:.1f}%):")
print(f"  2025 border-county revenue: ${cowa_border_rev:,.0f}")
print(f"  Attributable to border effect: ${cowa_attributable:,.0f}")
print(f"  Share of border revenue: {cowa_attributable/cowa_border_rev*100:.1f}%")
print(f"  Share of CO+WA combined total revenue: {cowa_attributable/cowa_state_total*100:.2f}%")
print(f"  Number of border counties: 12 (6 CO + 6 WA)")
print(f"  Attributable $ per border county: ${cowa_attributable/12:,.0f}")

print(f"\nMoratorium regime (Montana, b={mt_b:.3f}, p={mt_p:.3f}, premium={mt_premium*100:.1f}%):")
print(f"  2025 border-county revenue: ${mt_border_rev:,.0f}")
print(f"  Attributable to border effect: ${mt_attributable:,.0f}")
print(f"  Share of border revenue: {mt_attributable/mt_border_rev*100:.1f}%")
print(f"  Share of MT total revenue: {mt_attributable/mt_state_total*100:.2f}%")
print(f"  Number of border counties: 8")
print(f"  Attributable $ per border county: ${mt_attributable/8:,.0f}")

print(f"\nRatio, capped:moratorium, raw dollars: {cowa_attributable/mt_attributable:.2f}x")
print(f"Ratio, capped:moratorium, share of border revenue: {(cowa_attributable/cowa_border_rev)/(mt_attributable/mt_border_rev):.2f}x")
print(f"Ratio, capped:moratorium, $ per border county: {(cowa_attributable/12)/(mt_attributable/8):.2f}x")

out = pd.DataFrame([
    dict(regime='Capped (CO+WA pooled)', coef=pooled_b, p=pooled_p, premium_pct=pooled_premium*100,
         border_rev=cowa_border_rev, state_total=cowa_state_total, attributable=cowa_attributable,
         share_border_pct=cowa_attributable/cowa_border_rev*100,
         share_state_pct=cowa_attributable/cowa_state_total*100,
         n_border_counties=12, attributable_per_county=cowa_attributable/12),
    dict(regime='Moratorium (Montana)', coef=mt_b, p=mt_p, premium_pct=mt_premium*100,
         border_rev=mt_border_rev, state_total=mt_state_total, attributable=mt_attributable,
         share_border_pct=mt_attributable/mt_border_rev*100,
         share_state_pct=mt_attributable/mt_state_total*100,
         n_border_counties=8, attributable_per_county=mt_attributable/8),
])
out.to_csv(os.path.join(TABLES, 'regime_transfer_comparison.csv'), index=False)
print(f"\nSaved {os.path.join(TABLES, 'regime_transfer_comparison.csv')}")
