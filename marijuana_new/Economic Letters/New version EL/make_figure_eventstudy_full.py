# ============================================================
# Rebuilt in New version EL/ -- see analyze_eventstudy_full.py in
# this same folder for the regression this figure visualizes.
# ============================================================
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#c1502e'; GRAY = '#8a8a8a'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

with open(os.path.join(DATA, 'eventstudy_full_results.json')) as f:
    res = json.load(f)

BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
xpos = np.array([-3, -2, -1, 0, 1, 2, 3])
labels = ['≤−3', '−2', '−1', '0\n(switch)', '+1', '+2', '≥+3']

fig, ax = plt.subplots(figsize=(8.2, 5.0))
for offset, (key, color, name) in enumerate([('raw', GRAY, 'Raw (no controls)'), ('adjusted', ACCENT, 'Mean-reversion-adjusted (preferred)')]):
    dx = -0.08 if offset == 0 else 0.08
    coefs = [res[key][b]['coef'] for b in BINS]
    ses = [res[key][b]['se'] for b in BINS]
    ax.errorbar(xpos + dx, coefs, yerr=[1.96 * se for se in ses], fmt='o', color=color, capsize=4, ms=9,
                elinewidth=1.8, label=name)

ax.axhline(0, color='#bbb', lw=1, linestyle='--')
ax.axvline(0, color='#eee', lw=10, zorder=0)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=8.5, frameon=False, loc='upper right')
healy(ax, title='Event Study: Treatment (Switcher) vs. Control (Never-Switcher) Counties',
      xlabel='Years relative to switch (t=0: county crosses the 50-mile legal-market threshold)',
      ylabel='Coefficient (overdose deaths per 100,000, vs. never-switcher counties)')
plt.tight_layout(rect=[0, 0.16, 1, 0.94])
fig.text(0.02, 0.02,
          'Reference/control group: counties that never cross the 50-mile threshold in 2020-2025 (n=1,779). Treatment group:\n'
          '436 eventual switchers, 177 of whose switch-year transitions are non-suppressed (et=0). Endpoint bins (≤-3, ≥+3) pool\n'
          'sparse tail event-times. Standard errors clustered by county (finer than the state-clustered SEs used elsewhere in this\n'
          'paper) -- under this clustering the joint test that all three leads equal zero is REJECTED (p=0.016, adjusted spec), a\n'
          'pre-trends signal that is weaker/passes under state-clustered SEs; reported here transparently, not smoothed over.',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig7_eventstudy_full.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig7_eventstudy_full.png")
