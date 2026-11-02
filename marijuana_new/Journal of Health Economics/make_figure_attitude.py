# ============================================================
# Standalone figure for the attitude-proxy check: "legalize
# marijuana" Google Trends search-interest index, pooled across
# 18 never-legal states (ND/WV dropped, zero pre-period search
# volume), pegged to each state's own reference switch year.
# Same style as make_figure_mechanisms.py's panel (d).
# ============================================================
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#9a7a4c'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

gt = pd.read_csv(os.path.join(DATA, 'google_trends_attitude_summary.csv'))
gt_raw = pd.read_csv(os.path.join(DATA, 'google_trends_attitude_summary_raw.csv'))

fig, ax = plt.subplots(figsize=(7.8, 5.3))
ax.errorbar(gt_raw['event_time'] - 0.06, gt_raw['mean'], yerr=1.96 * gt_raw['se'].fillna(0), fmt='o',
            color='#b8b8b8', capsize=4, ms=8, elinewidth=1.5, label='Raw (not detrended)')
ax.errorbar(gt['event_time'] + 0.06, gt['mean'], yerr=1.96 * gt['se'].fillna(0), fmt='D',
            color=ACCENT, capsize=4, ms=9, elinewidth=1.9, label='Detrended (net of national search trend)')
ax.axhline(1, color='#bbb', lw=1, linestyle='--')
ax.axvline(0, color='#eee', lw=10, zorder=0)
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xticklabels(['−3', '−2', '−1', '0\n(ref. switch)', '+1', '+2', '+3'], fontsize=9)
healy(ax, title='Figure: "Legalize Marijuana" Search-Interest Index, Border States',
      xlabel='Years relative to state\'s reference switch year',
      ylabel='Search index (pre-switch mean = 1.0)')
ax.legend(loc='upper right', fontsize=8, frameon=False)
plt.tight_layout(rect=[0, 0.20, 1, 0.90])
note = ('Pooled across 18 never-legal border states (ND, WV dropped -- zero pre-period search volume for this exact\n'
        'phrase). State-level descriptive index, not a clustered regression -- Google Trends\' finest reliable geography\n'
        'is state, so this does NOT use the county-level 50-mile distance/switcher design used elsewhere in this project.\n'
        'National US search volume for "legalize marijuana" fell ~15%/year over 2019-2025 (36 to 13 on Trends\' own\n'
        'scale) -- the RAW series (grey) mostly just recreates this common trend, since post-periods mechanically fall\n'
        'in later calendar years. The DETRENDED series (gold, divides by the national value each year first) removes\n'
        'most of this: et=0 and et=+2 are no longer distinguishable from the pre-period baseline. A real dip remains at\n'
        'et=+1 even after detrending, but et=+3 rests on only 4 state-years -- too thin to interpret.')
fig.text(0.02, 0.01, note, fontsize=7.0, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig_attitude_google_trends.png'), dpi=170, facecolor='white')
plt.close()
print('Saved fig_attitude_google_trends.png')
