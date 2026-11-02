# ============================================================
# Comparison figure: full-sample (2020-2025, includes COVID years)
# vs. COVID-years-dropped (2022-2025 only) event study for the main
# overdose-mortality design. Same Kieran Healy style as
# make_figures_eventstudy.py.
# ============================================================
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; FULL_COLOR = '#8a8a8a'; NOCOVID_COLOR = '#c1502e'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

with open(os.path.join(DATA, 'eventstudy_main_results.json')) as f:
    full = json.load(f)['adjusted']
with open(os.path.join(DATA, 'eventstudy_main_nocovid_results.json')) as f:
    nocovid = json.load(f)['adjusted']

fig, ax = plt.subplots(figsize=(7.8, 5.2))

full_bins = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
full_x = np.array([-3, -2, -1, 0, 1, 2, 3])
nocovid_bins = ['et_m1', 'et_0', 'et_p1', 'et_p2']
nocovid_x = np.array([-1, 0, 1, 2]) + 0.08  # slight offset so overlapping points/CIs are readable

for x, b in zip(full_x, full_bins):
    c, se = full[b]['coef'], full[b]['se']
    ax.errorbar(x, c, yerr=1.96 * se, fmt='o', color=FULL_COLOR, capsize=4, ms=8, elinewidth=1.6,
                label='Full sample, 2020-2025 (incl. COVID years)' if x == -3 else None)

for x, b in zip(nocovid_x, nocovid_bins):
    c, se = nocovid[b]['coef'], nocovid[b]['se']
    ax.errorbar(x, c, yerr=1.96 * se, fmt='D', color=NOCOVID_COLOR, capsize=4, ms=8, elinewidth=1.9,
                label='COVID years (2020-2021) dropped' if x == -1 + 0.08 else None)

ax.axhline(0, color='#bbb', lw=1, linestyle='--')
ax.axvline(0, color='#eee', lw=10, zorder=0)
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xticklabels(['−3', '−2', '−1', '0\n(switch)', '+1', '+2', '+3'], fontsize=9)
healy(ax, title='Figure: Event Study, Overdose Mortality -- With vs. Without COVID Years (2020-2021)',
      xlabel='Years relative to switch (t=0: county crosses the 50-mile threshold)',
      ylabel='Coefficient (overdose deaths per 100,000, vs. far-away control counties)')
ax.legend(loc='upper left', fontsize=8, frameon=False)
plt.tight_layout(rect=[0, 0.13, 1, 0.90])
note = ('No-COVID spec uses transitions 2022-2023/2023-2024/2024-2025 only (n=397 switchers, 1,779 far-away controls);\n'
        'et_m2/et_m3 unavailable -- no switcher in this window has a switch year late enough to observe those pre-periods.\n'
        'Both specs: baseline-rate-adjusted, standard errors clustered by county.')
fig.text(0.02, 0.02, note, fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig_eventstudy_main_covid_comparison.png'), dpi=170, facecolor='white')
plt.close()
print('Saved fig_eventstudy_main_covid_comparison.png')
