# ============================================================
# Two-panel figure: how much 2025 border-county cannabis revenue is
# attributable to cross-border demand, capped regime (CO+WA pooled,
# Table 1 coefficient) vs moratorium regime (Montana, its own
# significant per-capita coefficient). Left panel: raw dollars.
# Right panel: dollars per border county, to separate "which regime
# captures more in total" from "which regime captures more per county
# exposed to the effect."
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

df = pd.read_csv(os.path.join(TABLES, 'regime_transfer_comparison.csv')).set_index('regime')
labels = ['Colorado + Washington\n(capped)', 'Montana\n(moratorium)']
colors = [PALETTE['blue'], PALETTE['gold']]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# ---------- Left: total dollars transferred ----------
ax = axes[0]
vals = df['attributable'] / 1e6
bars = ax.bar([0, 1], vals, color=colors, width=0.55)
ax.set_ylim(0, vals.max() * 1.3)
for i, (regime, row) in enumerate(df.iterrows()):
    ax.text(i, row['attributable'] / 1e6 + vals.max() * 0.03,
             f"${row['attributable']/1e6:,.1f}M\n({row['share_border_pct']:.0f}% of border revenue)",
             ha='center', fontsize=9.5, color=PALETTE['dark'])
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylabel('Revenue transferred, 2025 ($ millions)')
ax.set_title('Total dollars transferred from\nnon-legal neighbors', loc='left', pad=12)

# ---------- Right: dollars per border county ----------
ax = axes[1]
vals2 = df['attributable_per_county'] / 1e6
ax.bar([0, 1], vals2, color=colors, width=0.55)
ax.set_ylim(0, vals2.max() * 1.3)
for i, (regime, row) in enumerate(df.iterrows()):
    ax.text(i, row['attributable_per_county'] / 1e6 + vals2.max() * 0.03,
             f"${row['attributable_per_county']/1e6:,.1f}M\n({int(row['n_border_counties'])} counties)",
             ha='center', fontsize=9.5, color=PALETTE['dark'])
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylabel('Revenue transferred per border county ($ millions)')
ax.set_title('Transferred per border county\n(intensity of capture)', loc='left', pad=12)

fig.suptitle('Fiscal transfer from non-legal-state neighbors, capped vs. moratorium regime, 2025',
             fontsize=12, fontweight='bold', y=1.04)
fig.text(0.5, -0.03,
         'Capped regime (CO+WA): pooled Table 1 coefficient, $b=0.341$, $p=0.033$. '
         'Montana: revenue-per-capita coefficient, $b=0.654$, $p=0.036$.\n'
         'Both figures are accounting decompositions of an already-estimated coefficient, not independent causal magnitudes.',
         ha='center', fontsize=8.5, color=PALETTE['gray'])
fig.tight_layout()

out_path = os.path.join(BASE, 'visualized_regime_transfer.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved {out_path}")
