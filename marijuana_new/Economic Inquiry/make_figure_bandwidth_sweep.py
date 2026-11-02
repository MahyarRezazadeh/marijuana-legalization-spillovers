# ============================================================
# Transparency figure: the continuous specification's coefficient
# and p-value across a full sweep of sample-restriction radii,
# with the CCT (Calonico-Cattaneo-Titiunik) MSE-optimal bandwidth
# marked. Shows honestly that ad hoc radii in the 150-200mi range
# cross p<0.05 while the formal, pre-specified optimal bandwidth
# (30.4mi around the threshold, ~80mi total cutoff) does not --
# i.e., the "fix" is not robust to a principled selection rule.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
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

panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel_new.csv'))
main = panel[panel['d_od'].notna()].copy()

def cluster_fit(y, X, groups):
    Xc = sm.add_constant(X)
    g = pd.Categorical(groups).codes
    return sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})

radii = list(range(500, 69, -10))
rows = []
for radius in radii:
    sub = main[(main['dist_2020_mi'] <= radius) | (main['dist_2023_mi'] <= radius)]
    if len(sub) < 30:
        continue
    r = cluster_fit(sub['d_od'], sub[['d_dist_mi']], sub['state_abbr'])
    rows.append(dict(radius=radius, n=len(sub), coef=r.params['d_dist_mi'],
                      se=r.bse['d_dist_mi'], p=r.pvalues['d_dist_mi']))
sweep = pd.DataFrame(rows)
sweep.to_csv(os.path.join(DATA, 'bandwidth_sweep.csv'), index=False)

CCT_RADIUS = 80  # 50mi threshold + 30.4mi CCT-optimal bandwidth, rounded

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.plot(sweep['radius'], sweep['coef'], color=INK, lw=1.8)
ax.fill_between(sweep['radius'], sweep['coef']-1.96*sweep['se'], sweep['coef']+1.96*sweep['se'],
                 color=INK, alpha=0.1, linewidth=0)
ax.axhline(0, color='#ccc', lw=0.8)
ax.axvline(CCT_RADIUS, color=ACCENT2, lw=1.6, linestyle='--', label='CCT-optimal bandwidth (~80mi)')
ax.legend(fontsize=7.5, frameon=False, loc='lower right')
healy(ax, title='(a) Coefficient vs. sample-restriction radius', xlabel='Radius (miles)', ylabel='Continuous-distance coefficient')

ax = axes[1]
colors = [ACCENT if p < 0.05 else GRAY for p in sweep['p']]
ax.scatter(sweep['radius'], sweep['p'], c=colors, s=22)
ax.axhline(0.05, color='#bbb', lw=1, linestyle=':')
ax.axvline(CCT_RADIUS, color=ACCENT2, lw=1.6, linestyle='--')
healy(ax, title='(b) p-value vs. sample-restriction radius', xlabel='Radius (miles)', ylabel='p-value')

fig.suptitle('Figure F. The Continuous-Distance Result Is Not Robust to a Principled Bandwidth Choice',
             fontsize=11, x=0.02, ha='left', y=0.99, color=INK)
caption = ('Radius = cutoff applied to dist_2020_mi or dist_2023_mi (counties included if either falls within it). '
           'Ad hoc radii around 150-200mi cross p<0.05 (red points, panel b), but the CCT (Calonico-Cattaneo-\n'
           'Titiunik) MSE-optimal bandwidth -- chosen by a pre-specified algorithm, not by which radius happens to '
           'be significant -- is only ~80mi total and gives a null result (p=0.82). We read the mid-range\n'
           'significant window as a multiple-comparisons artifact of the sweep, not a robust finding.')
plt.tight_layout(rect=[0, 0.12, 1, 0.90])
fig.text(0.02, 0.02, caption, fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figF_bandwidth_sweep.png'), dpi=170, facecolor='white')
plt.close()
print("Saved figF_bandwidth_sweep.png")
print(sweep.to_string(index=False))
