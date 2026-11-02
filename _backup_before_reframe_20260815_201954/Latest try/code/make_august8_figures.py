# ============================================================
# Regenerates the figures needed for August_8th.tex in a clean,
# Kieran-Healy-inspired style (minimal gridlines, no top/right
# spines, muted palette, direct labeling). Reuses already-verified
# underlying data/coefficients -- no new estimation, only re-plotting.
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from healy_style import apply_healy_style, PALETTE

apply_healy_style()

PROJ = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\entry game-marijuana"
OUT = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\Latest try\figures"
os.makedirs(OUT, exist_ok=True)

# ---------- Figure 1: revenue per store, border vs interior (levels) ----------
panel = pd.read_csv(os.path.join(PROJ, 'tables', 'revenue_perstore_border70_panel.csv'), dtype={'fips5': str})
grp = panel.groupby('border_70mi')['revenue_per_store'].mean() / 1e6

fig, ax = plt.subplots(figsize=(5.2, 4.2))
labels = ['Interior', 'Border\n($\\leq$70 mi)']
vals = [grp[0], grp[1]]
colors = [PALETTE['gray'], PALETTE['red']]
bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"${v:.2f}M", ha='center', va='bottom',
            fontsize=10, color='#2B2B2B')
ax.set_ylabel('Mean revenue per dispensary ($ millions/year)')
ax.set_title('Revenue per dispensary is higher\nin border counties')
ax.grid(axis='x', visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'a8_fig1_revenue_levels.png'))
plt.close(fig)
print('Saved a8_fig1_revenue_levels.png')

# ---------- Figure 2: dose-response, continuous distance (binned scatter) ----------
dose_panel = pd.read_csv(os.path.join(PROJ, 'tables', 'revenue_perstore_captivepop_panel.csv'), dtype={'fips5': str})
controls = 'log_population + C(fy_year) + C(state_abbr)'
d = dose_panel.dropna(subset=['log_dist_nonlegal', 'log_revenue_per_store']).copy()
y_resid = smf.ols(f'log_revenue_per_store ~ {controls}', data=d).fit().resid
x_resid = smf.ols(f'log_dist_nonlegal ~ {controls}', data=d).fit().resid
d['y_resid'], d['x_resid'] = y_resid, x_resid
d['bin'] = pd.qcut(d['x_resid'], q=10, duplicates='drop')
binned = d.groupby('bin', observed=True).agg(x_mean=('x_resid', 'mean'), y_mean=('y_resid', 'mean'),
                                              n=('x_resid', 'size')).reset_index()
slope, intercept = np.polyfit(d['x_resid'], d['y_resid'], 1)
xs = np.linspace(d['x_resid'].min(), d['x_resid'].max(), 100)

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.scatter(binned['x_mean'], binned['y_mean'], s=binned['n'] * 2.2, color=PALETTE['dark'], zorder=3,
           edgecolor='white', linewidth=0.5)
ax.plot(xs, slope * xs + intercept, color=PALETTE['red'], linewidth=1.8, zorder=2)
ax.axhline(0, color='#B0B0B0', linewidth=0.6, zorder=1)
ax.axvline(0, color='#B0B0B0', linewidth=0.6, zorder=1)
ax.set_xlabel('log(distance to nearest non-legal county), residualized')
ax.set_ylabel('log(revenue per store), residualized')
ax.set_title('Revenue per dispensary falls smoothly\nwith distance to the border')
ax.annotate(f'slope = {slope:+.3f}\n(HC1 $p$ = 0.001)', xy=(0.66, 0.86), xycoords='axes fraction',
            fontsize=9.5, color=PALETTE['red'])
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'a8_fig2_doseresponse.png'))
plt.close(fig)
print(f'Saved a8_fig2_doseresponse.png (slope={slope:+.4f})')

# ---------- Figure 3: hazard ratio by entrant rank ----------
ranks = [1, 2, 3, 4]
hr = [0.974, 0.696, 0.848, 0.895]
se_log = [0.1065, 0.1236, 0.1487, 0.1624]
coef = [np.log(h) for h in hr]
lo = [np.exp(c - 1.96 * s) for c, s in zip(coef, se_log)]
hi = [np.exp(c + 1.96 * s) for c, s in zip(coef, se_log)]

fig, ax = plt.subplots(figsize=(5.8, 4.4))
ax.errorbar(ranks, hr, yerr=[[h - l for h, l in zip(hr, lo)], [u - h for h, u in zip(hr, hi)]],
            fmt='o', color=PALETTE['red'], ecolor=PALETTE['red'], capsize=4, markersize=7, zorder=3)
ax.plot(ranks, hr, color=PALETTE['red'], linewidth=1, alpha=0.5, zorder=2)
ax.axhline(1.0, color='#B0B0B0', linewidth=0.8, linestyle='--', zorder=1)
ax.text(4.05, 1.02, 'no effect', fontsize=9, color='#7C7C7C')
ax.set_xticks(ranks)
ax.set_xticklabels(['1st', '2nd', '3rd', '4th'])
ax.set_xlabel('Entrant rank')
ax.set_ylabel('Hazard ratio, log(distance to border)')
ax.set_title('The distance effect on entry speed\nis specific to the second entrant')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'a8_fig3_hazard_by_rank.png'))
plt.close(fig)
print('Saved a8_fig3_hazard_by_rank.png')

# ---------- Figure 4: theoretical model illustration ----------
# Hazard_2(d) = p(d) * mu_H, with p(d) a logistic-shaped probability of
# high cross-border demand type, decreasing in distance d.
d_grid = np.linspace(0, 150, 300)
d_bar = 45  # illustrative "reach" parameter
p_d = 1 / (1 + np.exp((d_grid - d_bar) / 12))
mu_H = 1.0
hazard = p_d * mu_H

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(d_grid, hazard, color=PALETTE['blue'], linewidth=2.2, zorder=3)
ax.axvline(70, color=PALETTE['gray'], linewidth=1, linestyle=':', zorder=1)
ax.text(72, hazard.max() * 0.92, '70-mile\nthreshold used\nin the data', fontsize=8.5, color=PALETTE['gray'])
ax.set_xlabel('Distance to nearest nonlegal county, $d_i$ (miles)')
ax.set_ylabel('Model-implied 2nd-entry hazard, $p(d_i)\\,\\mu_H$')
ax.set_title('Model prediction: 2nd-entry hazard\nfalls with distance to the border')
ax.set_ylim(0, 1.08)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'a8_fig4_model_hazard.png'))
plt.close(fig)
print('Saved a8_fig4_model_hazard.png')
