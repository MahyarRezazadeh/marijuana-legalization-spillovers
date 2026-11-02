# ============================================================
# Combined figure for all four mechanism checks (not part of the
# main paper -- diagnostic/exploratory only), single 2x2 grid:
# (a) hard-drug possession arrests, (b) hard-drug sale/mfg arrests,
# (c) cannabis tax revenue grand-opening index, (d) naloxone search
# interest index.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=7.5, colors=INK)
    if title:  ax.set_title(title, fontsize=8.5, loc='left', pad=5, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=7.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7.5, color=MUTED)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))

BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
xpos_all = [-3, -2, -1, 0, 1, 2, 3]

def plot_eventstudy(ax, results_file, color, title, ylabel):
    with open(os.path.join(DATA, results_file)) as f:
        res = json.load(f)
    adj = res['adjusted']
    xs = [x for x, b in zip(xpos_all, BINS) if b in adj]
    coefs = [adj[b]['coef'] for b in BINS if b in adj]
    ses = [adj[b]['se'] for b in BINS if b in adj]
    ax.errorbar(xs, coefs, yerr=[1.96 * se for se in ses], fmt='o', color=color, capsize=3, ms=7, elinewidth=1.5)
    ax.axhline(0, color='#bbb', lw=1, linestyle='--')
    ax.axvline(0, color='#eee', lw=8, zorder=0)
    ax.set_xticks(xs)
    healy(ax, title=title, xlabel='Years relative to switch', ylabel=ylabel)

plot_eventstudy(axes[0, 0], 'eventstudy_harddrug_results.json', '#4c7a4c',
                 '(a) Hard-drug POSSESSION arrests (157+159)', 'Coef. (per 100,000)')
plot_eventstudy(axes[0, 1], 'eventstudy_harddrugsale_results.json', '#7a4c4c',
                 '(b) Hard-drug SALE/MFG arrests (152+154)', 'Coef. (per 100,000)')

go = pd.read_csv(os.path.join(DATA, 'grandopening_summary.csv'))
ax = axes[1, 0]
ax.errorbar(go['event_quarter'], go['mean'], yerr=1.96 * go['se'].fillna(0), fmt='o-', color='#4c6a9a', capsize=3, ms=6, elinewidth=1.3)
ax.axhline(1, color='#bbb', lw=1, linestyle='--')
ax.axvline(0, color='#eee', lw=8, zorder=0)
healy(ax, title='(c) Cannabis tax revenue index (legal states)', xlabel='Quarters relative to retail opening',
      ylabel='Revenue index (pre-opening mean = 1.0)')

gt = pd.read_csv(os.path.join(DATA, 'google_trends_summary.csv'))
ax = axes[1, 1]
ax.errorbar(gt['event_time'], gt['mean'], yerr=1.96 * gt['se'].fillna(0), fmt='o-', color='#9a7a4c', capsize=3, ms=6, elinewidth=1.3)
ax.axhline(1, color='#bbb', lw=1, linestyle='--')
ax.axvline(0, color='#eee', lw=8, zorder=0)
healy(ax, title='(d) "Naloxone" search-interest index (21 states)', xlabel='Years relative to reference switch year',
      ylabel='Search index (pre-switch mean = 1.0)')

plt.tight_layout(rect=[0, 0.06, 1, 0.94])
fig.suptitle('Mechanism Checks (Exploratory, Not in Main Paper)', fontsize=11, x=0.02, ha='left', y=0.98)
fig.text(0.02, 0.01,
          '(a)-(b): same treatment/control/threshold and specification as Figure 1, county-clustered. (c)-(d): descriptive,\n'
          'no control group -- (c) rules out a spike-then-fade pattern (revenue keeps growing); (d) shows a peak-then-decline\n'
          'shape but cannot rule out a confounding national naloxone-attention trend over the same period.',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'exploratory_mechanism_checks.png'), dpi=170, facecolor='white')
plt.close()
print("Saved exploratory_mechanism_checks.png")
