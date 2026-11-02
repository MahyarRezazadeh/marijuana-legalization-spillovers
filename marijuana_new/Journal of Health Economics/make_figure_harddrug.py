# ============================================================
# Journal of Health Economics draft -- hard-drug possession arrest
# event-study figure, same Healy style and same +/-3 window as
# Figures 1-2, for direct visual comparison with the overdose-
# mortality event study.
# ============================================================
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#4c7a4c'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

with open(os.path.join(DATA, 'eventstudy_harddrug_results.json')) as f:
    res = json.load(f)
adj = res['adjusted']

BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
xpos_all = [-3, -2, -1, 0, 1, 2, 3]
labels_all = ['≤−3', '−2', '−1', '0\n(switch)', '+1', '+2', '≥+3']

xs = [x for x, b in zip(xpos_all, BINS) if b in adj]
labels = [l for l, b in zip(labels_all, BINS) if b in adj]
coefs = [adj[b]['coef'] for b in BINS if b in adj]
ses = [adj[b]['se'] for b in BINS if b in adj]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.errorbar(xs, coefs, yerr=[1.96 * se for se in ses], fmt='o', color=ACCENT, capsize=4, ms=9, elinewidth=1.8)
ax.axhline(0, color='#bbb', lw=1, linestyle='--')
ax.axvline(0, color='#eee', lw=10, zorder=0)
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
healy(ax, title='Figure 5. Event Study: Hard-Drug (Opioid/Cocaine + Synthetic-Narcotic) Possession Arrests',
      xlabel='Years relative to switch (t=0: county crosses the 50-mile threshold)',
      ylabel='Coefficient (hard-drug possession arrests per 100,000)')
plt.tight_layout(rect=[0, 0.16, 1, 0.90])
fig.text(0.02, 0.02,
          f"Outcome: FBI CDE codes 157 (opium/cocaine possession) + 159 (synthetic-narcotic possession), county-level,\n"
          f"crosswalked from agency-level arrest data. Same treatment/control/threshold definition and specification as\n"
          f"Figure 1 (baseline-2020-rate control, transition FE, county-clustered SEs). N={res['n']}.",
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig5_harddrug_eventstudy.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig5_harddrug_eventstudy.png")
