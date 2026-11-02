# ============================================================
# RDD figure: binned-scatter of overdose rate against signed
# distance to the state border (negative = never-legal side,
# positive = legal side), with a local-linear fit each side,
# for 2020 and 2023 separately -- Kieran Healy style.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'
MUTED = '#8a8a8a'
ACCENT = '#c1502e'
ACCENT2 = '#4c9a6a'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

panel = pd.read_csv(os.path.join(DATA, 'rdd_overdose_panel.csv'))
bw_map = {2020: 130, 2023: 80}  # CCT-optimal bandwidths from run_rdd_overdose.py

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, year, col in zip(axes, [2020, 2023], [ACCENT2, ACCENT]):
    bw = bw_map[year]
    sub = panel[(panel['year'] == year) & panel['od_rate'].notna() & (panel['X'].abs() <= bw)].copy()

    # binned scatter (10-mile bins)
    sub['bin'] = (sub['X'] // 10) * 10 + 5
    binned = sub.groupby('bin')['od_rate'].mean().reset_index()
    ax.scatter(binned['bin'], binned['od_rate'], s=22, color=MUTED, alpha=0.8, zorder=3)

    for side, xr in [('never_legal', sub[sub['X'] < 0]), ('legal', sub[sub['X'] >= 0])]:
        if len(xr) < 5:
            continue
        z = np.polyfit(xr['X'], xr['od_rate'], 1)
        xs = np.linspace(xr['X'].min(), xr['X'].max(), 30)
        ax.plot(xs, np.polyval(z, xs), color=col, lw=2.2, zorder=4)

    ax.axvline(0, color='#bbb', lw=1, linestyle='--')
    healy(ax, title=f'{year} (CCT-optimal bandwidth = {bw}mi)',
          xlabel='Signed distance to border (mi); negative = never-legal side',
          ylabel='Overdose rate per 100,000' if year == 2020 else '')
fig.suptitle('Figure 6. Spatial RDD: Overdose Rate at the Legal-Market Border, 2020 vs. 2023',
             fontsize=10.5, x=0.02, ha='left', y=1.03, color=INK)
fig.text(0.02, -0.05,
         'Points are 10-mile-bin averages. Lines are separate local-linear fits each side of the border '
         '(cutoff at 0). Diff-in-disc = (2023 jump) - (2020 jump) = +3.82 (se=12.18, p=0.754), not significant.',
         fontsize=7, color=MUTED, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig6_rdd_border.png'), dpi=170, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig6_rdd_border.png")
