# ============================================================
# Per-state fiscal-spillover accounting, extending the pooled CO+WA
# calculation already in the AE draft appendix (b=0.341 -> ~$101M)
# to each state individually, using whichever margin is actually
# significant for that state's regime (revenue per dispensary for
# capped CO/WA, revenue per capita for moratorium-constrained MT).
#
# Attributable-revenue formula, same logic as the existing appendix
# calculation: if actual = counterfactual * (1 + premium), then
# attributable = actual * premium / (1 + premium), where premium is
# the exact log-coefficient transform 100*(exp(b)-1).
#
# This is an ACCOUNTING DECOMPOSITION of an already-estimated
# coefficient, not an independent causal magnitude -- same caveat as
# the existing appendix language, carried through here.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

# ---------- Regression coefficients already estimated/verified this session ----------
coefs = {
    'CO': dict(regime='Capped (local-option bans, no statewide number)', margin='Revenue per dispensary',
               b=0.552, p=0.005, sig=True, entry_b=None),
    'WA': dict(regime='Capped (statewide numerical cap, 334->556)', margin='Revenue per dispensary',
               b=0.233, p=0.334, sig=False, entry_b=None),
    'MT': dict(regime='Moratorium (fixed pool of 2020 medical licensees)', margin='Revenue per capita',
               b=0.654, p=0.036, sig=True, entry_b=(0.142, 0.106)),
}
# pooled CO+WA numbers, already in the manuscript appendix, kept for comparison
pooled_cowa = dict(b=0.341, p=0.033)
entry_cowa = dict(b=-0.054, p=0.457)  # rerun_entry_density_corrected.py, this session

# ---------- 2025 border-county revenue, by state ----------
perstore = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel_CORRECTED.csv'), dtype={'fips5': str})
p2025 = perstore[perstore['fy_year'] == 2025]

mt = pd.read_csv(os.path.join(TABLES, 'montana_revenue_perstore_cross.csv'), dtype={'fips5': str})

rows = []
for st in ['CO', 'WA']:
    sub = p2025[p2025['state_abbr'] == st]
    border_rev = sub.loc[sub['border_70mi'] == 1, 'total_sales'].sum()
    interior_rev = sub.loc[sub['border_70mi'] == 0, 'total_sales'].sum()
    n_border = sub.loc[sub['border_70mi'] == 1, 'fips5'].nunique()
    n_interior = sub.loc[sub['border_70mi'] == 0, 'fips5'].nunique()
    rows.append(dict(state=st, border_rev_2025=border_rev, interior_rev_2025=interior_rev,
                      n_border=n_border, n_interior=n_interior))

mt_border_rev = mt.loc[mt['border_70mi'] == 1, 'total_sales'].sum()
mt_interior_rev = mt.loc[mt['border_70mi'] == 0, 'total_sales'].sum()
rows.append(dict(state='MT', border_rev_2025=mt_border_rev, interior_rev_2025=mt_interior_rev,
                  n_border=int(mt['border_70mi'].sum()), n_interior=int((mt['border_70mi'] == 0).sum())))

rev_df = pd.DataFrame(rows).set_index('state')
print("2025 border/interior revenue by state:")
print(rev_df)

# ---------- Attributable revenue ----------
print("\n" + "=" * 70)
print("FISCAL SPILLOVER, per state")
print("=" * 70)
out_rows = []
for st, c in coefs.items():
    premium = (np.exp(c['b']) - 1)
    total_state_2025 = rev_df.loc[st, 'border_rev_2025'] + rev_df.loc[st, 'interior_rev_2025']
    attributable = rev_df.loc[st, 'border_rev_2025'] * (premium / (1 + premium))
    share_of_border = premium / (1 + premium)
    share_of_state_total = attributable / total_state_2025
    print(f"\n{st}: margin={c['margin']}, b={c['b']:+.3f} (p={c['p']:.3f}, "
          f"{'significant' if c['sig'] else 'NOT significant -- illustrative only'})")
    print(f"  implied premium: {premium*100:+.1f}%")
    print(f"  2025 border-county revenue: ${rev_df.loc[st,'border_rev_2025']:,.0f}")
    print(f"  attributable to border effect: ${attributable:,.0f} "
          f"({share_of_border*100:.1f}% of border revenue, {share_of_state_total*100:.1f}% of state total)")
    out_rows.append(dict(state=st, regime=c['regime'], margin=c['margin'], coef=c['b'], p=c['p'],
                          sig=c['sig'], premium_pct=premium*100,
                          border_rev_2025=rev_df.loc[st, 'border_rev_2025'],
                          state_total_2025=total_state_2025,
                          attributable_usd=attributable,
                          share_of_border_rev_pct=share_of_border*100,
                          share_of_state_total_pct=share_of_state_total*100,
                          entry_density_b=c['entry_b'][0] if c['entry_b'] else entry_cowa['b'],
                          entry_density_p=c['entry_b'][1] if c['entry_b'] else entry_cowa['p']))

out = pd.DataFrame(out_rows)
out.to_csv(os.path.join(TABLES, 'fiscal_spillover_by_state.csv'), index=False)
print(f"\nSaved {os.path.join(TABLES, 'fiscal_spillover_by_state.csv')}")

# pooled CO+WA cross-check against existing appendix number
pooled_border_rev = rev_df.loc[['CO', 'WA'], 'border_rev_2025'].sum()
pooled_premium = np.exp(pooled_cowa['b']) - 1
pooled_attributable = pooled_border_rev * (pooled_premium / (1 + pooled_premium))
print(f"\nPooled CO+WA cross-check (existing appendix used b={pooled_cowa['b']}, ~$101M): "
      f"border rev=${pooled_border_rev:,.0f}, attributable=${pooled_attributable:,.0f}")
