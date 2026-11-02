import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

BLUE = '#1f77b4'; RED = '#d62728'; GRAY = '#7f7f7f'; GREEN = '#2ca02c'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8, colors='#444444')
    if title:  ax.set_title(title, fontsize=9, loc='left', pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color='#555555')
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color='#555555')

panel = pd.read_csv(os.path.join(DATA, 'partD_county_panel.csv'))

# Fig D0: treatment map
fig, ax = plt.subplots(figsize=(8, 5))
ctrl = panel[panel['switcher'] == 0].dropna(subset=['lat', 'lon'])
sw = panel[panel['switcher'] == 1].dropna(subset=['lat', 'lon'])
ax.scatter(ctrl['lon'], ctrl['lat'], c=GRAY, s=6, alpha=0.35, linewidths=0, label='Control (>=50 mi)')
ax.scatter(sw['lon'], sw['lat'], c=RED, s=22, alpha=0.85, linewidths=0, label='Switcher (became <50 mi)')
ax.legend(fontsize=8, frameon=False, loc='lower right')
healy(ax, title='Figure D0. Treatment Assignment by Distance to Border', xlabel='Longitude', ylabel='Latitude')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figD0_map.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig D1: coefficient plot, primary dist_100mi spec, all three outcomes (standardized for comparability)
import json
with open(os.path.join(DATA, 'partD_results.json')) as f:
    res = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
specs = [('d_zhvi_log__switcher', 'd_zhvi_log__dist_thresh', 'd_zhvi_log__log_dist', 'Home Value (log)', BLUE),
         ('d_zori_log__switcher', 'd_zori_log__dist_thresh', 'd_zori_log__log_dist', 'Rent (log)', GREEN),
         ('d_od__switcher', 'd_od__dist_thresh', 'd_od__log_dist', 'Overdose Rate / 100k', RED)]
for ax, (k_sw, k_dist, k_log, label, col) in zip(axes, specs):
    labels_y = ['Switcher', 'Dist., per\n50 mi', 'Log-dist.\n(robust.)']
    keys = [k_sw, k_dist, k_log]
    coefs, los, his = [], [], []
    for k in keys:
        b, se, p, n = res[k]
        coefs.append(b); los.append(b - 1.96*se); his.append(b + 1.96*se)
    y = np.arange(len(labels_y))
    for i in range(len(labels_y)):
        ax.errorbar(coefs[i], y[i], xerr=[[coefs[i]-los[i]], [his[i]-coefs[i]]],
                    fmt='o', color=col, capsize=4, ms=7, elinewidth=1.6)
    ax.axvline(0, color='#999', lw=1, linestyle='--')
    ax.set_yticks(y); ax.set_yticklabels(labels_y, fontsize=8)
    ax.invert_yaxis()
    healy(ax, title=label)
fig.suptitle('Figure D1. Effect on Home Values, Rent, and Overdose Mortality (95% CI)', fontsize=9, x=0.02, ha='left', y=1.04)
fig.text(0.02, -0.04, 'SE clustered by state. First-difference (2020->2023). Overdose panel restricted to non-suppressed counties (see Section 2.3).',
         fontsize=6.5, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figD1_coefplot.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig D2: continuous treatment scatter for overdose (the significant-ish outcome) and ZHVI
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, ycol, label, col in [(axes[0], 'd_zhvi_log', 'Home Value (log change)', BLUE),
                              (axes[1], 'd_od', 'Overdose Rate Change / 100k', RED)]:
    x = panel['d_dist_thresh']
    y = panel[ycol]
    ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=9, alpha=0.35, color=col, linewidths=0)
    z = np.polyfit(x[ok], y[ok], 1)
    xr = np.linspace(x[ok].quantile(0.02), x[ok].quantile(0.98), 50)
    ax.plot(xr, np.polyval(z, xr), color='black', lw=1.8)
    ax.axvline(0, color='#ccc', lw=0.8); ax.axhline(0, color='#ccc', lw=0.8)
    healy(ax, title=label, xlabel='Change in distance to border (50 mi units)', ylabel='Change in outcome')
fig.suptitle('Figure D2. The Continuous-Treatment Regression: Distance Change vs. Outcome Change', fontsize=9, x=0.02, ha='left', y=1.03)
fig.text(0.02, -0.02, 'Negative x = county became closer to a newly-legal border.', fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figD2_treatment_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig D3: switcher vs control trend lines for overdose rate (the significant switcher result)
fig, ax = plt.subplots(figsize=(6, 4.5))
od_avail = panel[panel['d_od'].notna()]
tr = od_avail.groupby('switcher')[['od_rate20', 'od_rate23']].mean()
for swv, col, ls in [(0, GRAY, '-'), (1, RED, '--')]:
    lab = 'Control' if swv == 0 else 'Switcher'
    ax.plot([2020, 2023], [tr.loc[swv, 'od_rate20'], tr.loc[swv, 'od_rate23']],
            color=col, linestyle=ls, marker='o', ms=8, lw=2.2, label=lab)
ax.set_xticks([2020, 2023])
ax.legend(fontsize=8, frameon=False)
healy(ax, title='Figure D3. Drug Overdose Death Rate: Switchers vs. Controls', ylabel='Deaths per 100,000 (12-month-ending)')
fig.text(0.02, -0.03, 'CDC VSRR provisional county-level data, non-suppressed counties only (N=879).', fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figD3_overdose_trends.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Figures saved.")
