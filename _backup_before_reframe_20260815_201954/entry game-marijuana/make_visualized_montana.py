# ============================================================
# Two-panel summary of the Montana Result 2 contrast: border vs
# interior county means for (1) revenue per capita, the 2022-2025
# panel where the border effect is significant (b=+0.654, p=0.036
# county-clustered, from analyze_montana_revenue.py /
# tables/montana_revenue_results.csv), and (2) revenue per dispensary,
# the 2025-only cross-section where the border effect is a precise
# null (b=+0.012, p=0.958 HC1, from build_montana_perstore.py /
# tables/montana_revenue_perstore_cross.csv). Bars are raw group
# means (not regression-adjusted), annotated with the regression
# coefficient/p-value from each script's actual output, so the
# reader sees both the raw pattern and the controlled result.
# Reuses the project's established "Kieran Healy" matplotlib style
# (Latest try/code/healy_style.py) for visual consistency with the
# other figures already built for this paper family.
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

percapita = pd.read_csv(os.path.join(TABLES, 'montana_revenue_percapita_panel.csv'))
perstore = pd.read_csv(os.path.join(TABLES, 'montana_revenue_perstore_cross.csv'))

pc_means = percapita.groupby('border_70mi')['revenue_per_capita'].mean()
ps_means = perstore.groupby('border_70mi')['revenue_per_store'].mean()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

labels = ['Interior', 'Border\n(≤70mi)']
x = np.arange(2)
colors = [PALETTE['gray'], PALETTE['blue']]

ax = axes[0]
vals = [pc_means[0], pc_means[1]]
ax.bar(x, vals, color=colors, width=0.55)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Revenue per capita ($)')
ax.set_title('Revenue per capita\n(2022–2025 panel)', loc='left')
for xi, v in zip(x, vals):
    ax.text(xi, v + max(vals) * 0.02, f'${v:,.0f}', ha='center', fontsize=9.5, color=PALETTE['dark'])
ax.text(0.5, -0.30, 'border coef. $+0.654$, $p=0.036$ (county-cl.)\nsignificant',
        transform=ax.transAxes, ha='center', fontsize=9, color=PALETTE['dark'])

ax = axes[1]
vals = [ps_means[0], ps_means[1]]
ax.bar(x, vals, color=colors, width=0.55)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Revenue per dispensary ($)')
ax.set_title('Revenue per dispensary\n(2025 cross-section only)', loc='left')
for xi, v in zip(x, vals):
    ax.text(xi, v + max(vals) * 0.02, f'${v:,.0f}', ha='center', fontsize=9.5, color=PALETTE['dark'])
ax.text(0.5, -0.30, 'border coef. $+0.012$, $p=0.958$ (HC1)\nnull, $n=23$, single year',
        transform=ax.transAxes, ha='center', fontsize=9, color=PALETTE['gray'])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{v:,.0f}'))

fig.suptitle('Montana: the border pattern shows up in per-capita revenue, not per-dispensary revenue',
             fontsize=11.5, fontweight='bold', y=1.03)
fig.tight_layout()

out_path = os.path.join(BASE, 'visualized_montana.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved {out_path}")
print("\nRaw group means used:")
print("Revenue per capita -- interior:", round(pc_means[0], 1), "border:", round(pc_means[1], 1))
print("Revenue per store  -- interior:", round(ps_means[0], 1), "border:", round(ps_means[1], 1))
