# ============================================================
# Event-study figure: shows the raw pre-trends problem (leads
# comparable in size to the switch effect) alongside the
# mean-reversion-corrected version (baseline overdose-level
# control), so the reader can see both the problem and the fix
# in one place. Kieran Healy style, consistent with the rest of
# the project.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#c1502e'; ACCENT2 = '#4c9a6a'; GRAY = '#8a8a8a'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

stacked = pd.read_csv(os.path.join(DATA, 'pretrends_panel_rec_v2.csv'), dtype={'fips5': str})
stacked['g'] = pd.Categorical(stacked['state_abbr']).codes

def fit(formula, data):
    m = smf.ols(formula, data=data).fit(cov_type='cluster', cov_kwds={'groups': data['g']})
    out = {}
    for term in ['lead2plus', 'lead1', 'switch0']:
        out[term] = (m.params[term], m.bse[term])
    return out

raw = fit('d_od ~ lead2plus + lead1 + switch0 + C(transition)', stacked)
sub = stacked.dropna(subset=['baseline_rate'])
adj = fit('d_od ~ lead2plus + lead1 + switch0 + baseline_rate + C(transition)', sub)

labels = ['2+ transitions\nbefore switching', '1 transition\nbefore switching', 'Switch\ntransition']
xpos = np.array([-2, -1, 0])

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for offset, (res, color, name) in enumerate([(raw, GRAY, 'Raw (no controls)'), (adj, ACCENT, 'Mean-reversion-adjusted')]):
    dx = -0.08 if offset == 0 else 0.08
    coefs = [res['lead2plus'][0], res['lead1'][0], res['switch0'][0]]
    ses = [res['lead2plus'][1], res['lead1'][1], res['switch0'][1]]
    ax.errorbar(xpos + dx, coefs, yerr=[1.96*se for se in ses], fmt='o', color=color, capsize=4, ms=9,
                elinewidth=1.8, label=name)

ax.axhline(0, color='#bbb', lw=1, linestyle='--')
ax.axvline(-0.5, color='#eee', lw=6, zorder=0)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=8.5, frameon=False, loc='lower left')
healy(ax, title='Figure 5. Pre-Trends Check: Raw vs. Mean-Reversion-Adjusted',
      ylabel='Coefficient (overdose deaths per 100,000)')
plt.tight_layout(rect=[0, 0.12, 1, 0.94])
fig.text(0.02, 0.02, 'Reference group: counties that never switch in the 2020-2025 window. Raw leads are nearly as large as the\n'
                     'switch effect itself (a pre-trends violation); controlling for each county\'s baseline (2020) overdose-rate level\n'
                     '(a mean-reversion correction) shrinks both leads toward zero while the switch effect remains significant.',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig5_eventstudy_pretrends.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig5_eventstudy_pretrends.png")
