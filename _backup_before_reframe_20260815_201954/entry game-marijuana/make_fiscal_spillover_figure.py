# ============================================================
# Two-panel "policy allocation" figure built on the fiscal-spillover
# accounting in build_fiscal_spillover_table.py.
#
# Left panel: the three margins (entry density, revenue/dispensary,
# revenue/capita) side by side for capped (CO+WA pooled) vs Montana's
# moratorium regime -- shows WHICH margin absorbs the border effect
# under each regime.
# Right panel: 2025 border-county revenue per state, split into
# baseline (counterfactual, no border effect) vs attributable-to-
# border-effect, using each state's own significant margin. Shows the
# DOLLAR size of the spillover and what share of border revenue it is.
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

sys.path.insert(0, os.path.join(BASE, '..', 'Latest try', 'code'))
from healy_style import apply_healy_style, PALETTE
apply_healy_style()

spill = pd.read_csv(os.path.join(TABLES, 'fiscal_spillover_by_state.csv')).set_index('state')

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# ---------- Left panel: which margin moves, by regime ----------
ax = axes[0]
margins = ['Entry density\n(dispensaries/1,000)', 'Revenue per\ndispensary', 'Revenue per\ncapita']
capped = [-0.054, 0.341, 0.046]      # pooled CO+WA, all three margins (this session + existing Table 1/placebo)
capped_p = [0.457, 0.033, 0.833]
montana = [0.142, 0.012, 0.654]      # Montana, all three margins (this session + existing Table 1/2)
montana_p = [0.106, 0.958, 0.036]

x = np.arange(len(margins))
w = 0.35
b1 = ax.bar(x - w/2, capped, width=w, color=PALETTE['blue'], label='Colorado + Washington (capped)')
b2 = ax.bar(x + w/2, montana, width=w, color=PALETTE['gold'], label='Montana (moratorium)')

for bars, pvals in [(b1, capped_p), (b2, montana_p)]:
    for bar, p in zip(bars, pvals):
        h = bar.get_height()
        marker = '*' if p < 0.05 else ''
        va = 'bottom' if h >= 0 else 'top'
        offset = 0.02 if h >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2, h + offset, f'{h:+.3f}{marker}',
                ha='center', va=va, fontsize=9, color=PALETTE['dark'])

ax.axhline(0, color=PALETTE['dark'], lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(margins)
ax.set_ylabel('Border coefficient (log points)')
ax.set_title('Which margin absorbs the border effect?', loc='left')
ax.legend(loc='upper left')
ax.text(0.02, -0.24, '* p < 0.05. Capped states: effect concentrated in revenue per dispensary.\n'
                      'Montana: effect concentrated in revenue per capita, with a marginal entry-density response.',
        transform=ax.transAxes, fontsize=8.5, color=PALETTE['gray'])

# ---------- Right panel: dollar spillover, baseline vs attributable ----------
ax = axes[1]
states = ['CO', 'WA', 'MT']
border_rev = spill.loc[states, 'border_rev_2025'] / 1e6
attributable = spill.loc[states, 'attributable_usd'] / 1e6
baseline = border_rev - attributable
sig = spill.loc[states, 'sig']

xs = np.arange(3)
ax.bar(xs, baseline, width=0.55, color=PALETTE['gray'], label='Baseline (counterfactual)')
bars_attr = ax.bar(xs, attributable, width=0.55, bottom=baseline, color=PALETTE['red'],
                    label='Attributable to border effect')

ax.set_ylim(0, border_rev.max() * 1.28)

for i, st in enumerate(states):
    total = border_rev[st]
    share = spill.loc[st, 'share_of_border_rev_pct']
    hatch_note = '' if sig[st] else ' (not sig.)'
    ax.text(i, total + border_rev.max() * 0.03, f'${total:,.0f}M\n{share:.0f}% attributable{hatch_note}',
            ha='center', fontsize=9, color=PALETTE['dark'])
    if not sig[st]:
        bars_attr[i].set_alpha(0.45)
        bars_attr[i].set_hatch('//')

ax.set_xticks(xs)
ax.set_xticklabels(['Colorado', 'Washington', 'Montana'])
ax.set_ylabel('2025 border-county revenue ($ millions)')
ax.set_title('Dollar value of the border-county fiscal spillover', loc='left', pad=12)
ax.legend(loc='upper right')
ax.text(0.02, -0.20, 'Washington bar is hatched: its border coefficient is not statistically\n'
                      'significant (p=0.33), so this split is illustrative only.',
        transform=ax.transAxes, fontsize=8.5, color=PALETTE['gray'])

fig.suptitle('Fiscal spillover and the license-cap channel: capped states concentrate the effect\n'
             'in existing stores, Montana spreads it toward entry',
             fontsize=11.5, fontweight='bold', y=1.06)
fig.tight_layout()

out_path = os.path.join(BASE, 'visualized_fiscal_spillover.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved {out_path}")
