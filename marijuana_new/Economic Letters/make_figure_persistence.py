# ============================================================
# Persistence-check figure: switch-year effect vs. subsequent-year
# (lag1, lag2+) effects, raw and mean-reversion-adjusted. Kieran
# Healy style, consistent with the rest of the project.
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

with open(os.path.join(DATA, 'persistence_results.json')) as f:
    res = json.load(f)

labels = ['Switch\ntransition', '1 transition\nafter switching', '2+ transitions\nafter switching']
xpos = np.array([0, 1, 2])

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for offset, (key, color, name) in enumerate([('raw', GRAY, 'Raw (no controls)'), ('adjusted', ACCENT, 'Mean-reversion-adjusted')]):
    dx = -0.08 if offset == 0 else 0.08
    coefs = [res[key]['switch0']['coef'], res[key]['lag1']['coef'], res[key]['lag2plus']['coef']]
    ses = [res[key]['switch0']['se'], res[key]['lag1']['se'], res[key]['lag2plus']['se']]
    ax.errorbar(xpos + dx, coefs, yerr=[1.96*se for se in ses], fmt='o', color=color, capsize=4, ms=9,
                elinewidth=1.8, label=name)

ax.axhline(0, color='#bbb', lw=1, linestyle='--')
ax.axvline(0.5, color='#eee', lw=6, zorder=0)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=8.5, frameon=False, loc='upper left')
healy(ax, title='Figure 6. Persistence Check: Does the Switcher Effect Continue After Switching?',
      ylabel='Coefficient (overdose deaths per 100,000)')
plt.tight_layout(rect=[0, 0.14, 1, 0.94])
fig.text(0.02, 0.02, 'Reference group: counties that never switch in the 2020-2025 window. Each point is a separate one-year\n'
                     'transition-level effect (not cumulative). The switch-transition effect does not continue in subsequent years --\n'
                     'a joint test rejects that lag effects equal the switch effect (p=0.006, adjusted spec).',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig6_persistence.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig6_persistence.png")
