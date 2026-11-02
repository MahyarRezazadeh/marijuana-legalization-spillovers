# ============================================================
# Updated version of build_regime_transfer_comparison.py, using the
# most defensible coefficient for each regime rather than the pooled
# coefficient for both:
#  - CO+WA: 2025-specific border coefficient (b=0.614, p=0.002),
#    well-powered given 12 border counties and a decade of data,
#    and a materially better estimate of the CURRENT premium than
#    the pooled 2015-2025 average (b=0.341), which was dragged down
#    by 8 early years where the effect wasn't yet significant.
#  - Montana: pooled 2022-2025 coefficient (b=0.654, p=0.036), since
#    Montana's own 2025-only coefficient is not significant (p=0.122,
#    only 8 border counties in one year) -- pooling across its short
#    history is the more reliable estimate here.
# Answers directly: of the total premium dollar value across both
# regimes, what share belongs to Montana vs. Colorado+Washington.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

cowa_peryear = pd.read_csv(os.path.join(TABLES, 'cowa_peryear_fiscal_spillover.csv'))
row_2025 = cowa_peryear[cowa_peryear['year'] == 2025].iloc[0]
cowa_b, cowa_p = row_2025['coef'], row_2025['p_county_cl']
cowa_border_rev = row_2025['actual_border_revenue']
cowa_attributable = row_2025['attributable']

spill = pd.read_csv(os.path.join(TABLES, 'fiscal_spillover_by_state.csv')).set_index('state')
mt_b, mt_p = spill.loc['MT', 'coef'], spill.loc['MT', 'p']
mt_border_rev = spill.loc['MT', 'border_rev_2025']
mt_attributable = spill.loc['MT', 'attributable_usd']

catchment = pd.read_csv(os.path.join(TABLES, 'catchment_normalized_transfer.csv')).set_index('regime')
cowa_catchment_pop = catchment.loc['Capped (CO+WA pooled)', 'catchment_population']
mt_catchment_pop = catchment.loc['Moratorium (Montana)', 'catchment_population']

total_attributable = cowa_attributable + mt_attributable
cowa_share = cowa_attributable / total_attributable
mt_share = mt_attributable / total_attributable

cowa_per_capita = cowa_attributable / cowa_catchment_pop
mt_per_capita = mt_attributable / mt_catchment_pop

print("=" * 70)
print("UPDATED PREMIUM SHARE: CO+WA (2025-specific coef.) vs. MT (pooled coef.)")
print("=" * 70)
print(f"\nCO+WA: b={cowa_b:.3f} (p={cowa_p:.4f}, 2025-specific), border rev=${cowa_border_rev:,.0f}, "
      f"attributable=${cowa_attributable:,.0f}")
print(f"Montana: b={mt_b:.3f} (p={mt_p:.4f}, pooled 2022-2025), border rev=${mt_border_rev:,.0f}, "
      f"attributable=${mt_attributable:,.0f}")
print(f"\nTotal premium dollars across both regimes: ${total_attributable:,.0f}")
print(f"  Share to CO+WA: {cowa_share*100:.1f}%  (${cowa_attributable:,.0f})")
print(f"  Share to Montana: {mt_share*100:.1f}%  (${mt_attributable:,.0f})")
print(f"\nPer catchment resident: CO+WA=${cowa_per_capita:.2f}  Montana=${mt_per_capita:.2f}  "
      f"(ratio {cowa_per_capita/mt_per_capita:.2f}x)")

out = pd.DataFrame([
    dict(regime='Colorado + Washington (capped)', coef=cowa_b, p=cowa_p, coef_source='2025-specific',
         border_revenue=cowa_border_rev, attributable=cowa_attributable, share_of_total_premium_pct=cowa_share*100,
         catchment_population=cowa_catchment_pop, per_catchment_resident=cowa_per_capita),
    dict(regime='Montana (moratorium)', coef=mt_b, p=mt_p, coef_source='pooled 2022-2025',
         border_revenue=mt_border_rev, attributable=mt_attributable, share_of_total_premium_pct=mt_share*100,
         catchment_population=mt_catchment_pop, per_catchment_resident=mt_per_capita),
])
out.to_csv(os.path.join(TABLES, 'premium_share_updated.csv'), index=False)
print(f"\nSaved {os.path.join(TABLES, 'premium_share_updated.csv')}")
