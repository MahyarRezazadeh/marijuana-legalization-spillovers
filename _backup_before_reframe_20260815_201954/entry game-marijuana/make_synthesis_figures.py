# ============================================================
# Two figures for synthesis.tex:
# Fig A: incidence, all three margins x three states (extends the
#        earlier two-regime version to show CO and WA separately,
#        not just pooled).
# Fig B: political-economy finding -- share of CO counties that ever
#        had an operating dispensary, border vs interior.
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

# ---------- Fig A: incidence by state, all three margins ----------
states = ['Colorado', 'Washington', 'Montana']
entry = [0.015, -0.002, 0.142]
entry_p = [0.889, 0.940, 0.106]
perstore = [0.552, 0.233, 0.012]
perstore_p = [0.005, 0.334, 0.958]
percapita = [-0.013, 0.341, 0.654]
percapita_p = [0.968, 0.121, 0.036]

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(3)
w = 0.26
colors = [PALETTE['gray'], PALETTE['blue'], PALETTE['red']]
groups = [('Entry density', entry, entry_p), ('Revenue per\ndispensary', perstore, perstore_p),
          ('Revenue per\ncapita', percapita, percapita_p)]

for i, (label, vals, pvals) in enumerate(groups):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, vals, width=w, color=colors[i], label=label)
    for xi, v, p in zip(x + offset, vals, pvals):
        marker = '*' if p < 0.05 else ''
        va = 'bottom' if v >= 0 else 'top'
        pad = 0.02 if v >= 0 else -0.02
        ax.text(xi, v + pad, f'{v:+.3f}{marker}', ha='center', va=va, fontsize=8, color=PALETTE['dark'])

ax.axhline(0, color=PALETTE['dark'], lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.set_ylabel('Border coefficient (log points)')
ax.set_ylim(-0.08, max(entry + perstore + percapita) * 1.28)
ax.set_title('Incidence of the border effect, all three margins, by state', loc='left', pad=30)
ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(0, -0.12), frameon=False, fontsize=9)
ax.text(0.01, 1.03, '* p < 0.05 (county-clustered where a panel; HC1 for single-year cross-sections)',
        transform=ax.transAxes, fontsize=8, color=PALETTE['gray'])
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'synthesis_fig_incidence.png'), dpi=200, bbox_inches='tight')
print("Saved synthesis_fig_incidence.png")
plt.close(fig)

# ---------- Fig B: political economy, CO local exclusion ----------
pe = pd.read_csv(os.path.join(TABLES, 'co_local_ban_political_economy.csv'))
rates = pe.groupby('border_70mi')['has_dispensary'].mean()

fig, ax = plt.subplots(figsize=(6, 5))
labels = ['Interior\n(n=50)', 'Border\n(n=14)']
vals = [rates[0] * 100, rates[1] * 100]
bars = ax.bar([0, 1], vals, color=[PALETTE['gray'], PALETTE['blue']], width=0.5)
for i, v in enumerate(vals):
    ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=11, color=PALETTE['dark'])
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylabel('Share of counties with any operating dispensary, 2015-2025')
ax.set_ylim(0, 100)
ax.set_title('Colorado border counties are LESS likely\nto ever have an operating dispensary', loc='left', pad=12)
fig.text(0.5, -0.05, 'b=-0.263, p=0.036 (+ log population). Opposite sign from the fiscal-capture hypothesis.',
         ha='center', fontsize=8.5, color=PALETTE['gray'])
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'synthesis_fig_politicaleconomy.png'), dpi=200, bbox_inches='tight')
print("Saved synthesis_fig_politicaleconomy.png")
