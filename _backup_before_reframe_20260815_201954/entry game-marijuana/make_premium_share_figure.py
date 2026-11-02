# ============================================================
# Updated premium-share figure: what share of the total border-effect
# dollar premium (across both regimes) goes to CO+WA vs Montana, using
# each regime's own best-supported coefficient (2025-specific for
# CO+WA, pooled 2022-2025 for Montana -- see build_premium_share_
# updated.py for why). Left panel: share of the total pie. Right
# panel: per-catchment-resident intensity, same two regimes.
# ============================================================
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

sys.path.insert(0, os.path.join(BASE, '..', 'Latest try', 'code'))
from healy_style import apply_healy_style, PALETTE
apply_healy_style()

df = pd.read_csv(os.path.join(TABLES, 'premium_share_updated.csv')).set_index('regime')
labels = ['Colorado + Washington\n(capped)', 'Montana\n(moratorium)']
colors = [PALETTE['blue'], PALETTE['gold']]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))

# ---------- Left: horizontal stacked share bar ----------
ax = axes[0]
cowa_share = df.loc['Colorado + Washington (capped)', 'share_of_total_premium_pct']
mt_share = df.loc['Montana (moratorium)', 'share_of_total_premium_pct']
ax.barh([0], [cowa_share], color=colors[0], height=0.5, label=labels[0])
ax.barh([0], [mt_share], left=[cowa_share], color=colors[1], height=0.5, label=labels[1])
ax.text(cowa_share/2, 0, f'{cowa_share:.0f}%\n${df.loc["Colorado + Washington (capped)","attributable"]/1e6:.0f}M',
        ha='center', va='center', fontsize=10.5, color='white', fontweight='bold')
ax.text(cowa_share + mt_share/2, 0, f'{mt_share:.0f}%\n${df.loc["Montana (moratorium)","attributable"]/1e6:.0f}M',
        ha='center', va='center', fontsize=10.5, color='white', fontweight='bold')
ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xlabel('Share of total border-effect premium ($187.2M combined)')
ax.set_title('Who captures the premium?', loc='left', pad=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
ax.spines['left'].set_visible(False)
ax.grid(False)

# ---------- Right: per catchment resident ----------
ax = axes[1]
vals = df['per_catchment_resident']
ax.bar([0, 1], [vals.iloc[0], vals.iloc[1]], color=colors, width=0.55)
ax.set_ylim(0, vals.max() * 1.3)
for i, v in enumerate(vals):
    ax.text(i, v + vals.max() * 0.03, f'${v:,.0f}', ha='center', fontsize=10, color=PALETTE['dark'])
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylabel('Premium $ per catchment resident')
ax.set_title('Intensity per potential\ncross-border customer', loc='left', pad=12)

fig.suptitle('The border-effect premium is concentrated in the capped regime,\nin total and per potential customer',
             fontsize=12, fontweight='bold', y=1.08)
fig.text(0.5, -0.12,
         'CO+WA uses its 2025-specific coefficient ($b=0.614$, $p=0.002$); Montana uses its pooled\n'
         '2022-2025 coefficient ($b=0.654$, $p=0.036$), since Montana\'s 2025-only estimate is not significant on its own.',
         ha='center', fontsize=8.5, color=PALETTE['gray'])
fig.tight_layout()

out_path = os.path.join(BASE, 'visualized_premium_share.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved {out_path}")
