# ============================================================
# Visual discontinuity check: does overdose mortality show a
# genuine break as a function of distance to the nearest
# legal-market county, and if so, where? Two figures:
#
#   Fig 1 ("scan"): binned mean overdose rate vs. raw distance,
#   0-250 miles, 10-mile bins, for 2020 (thinnest legal-market
#   coverage) and 2025 (widest) -- with 50/70/100-mile reference
#   lines, the three thresholds used elsewhere in this paper.
#   Purpose: eyeball whether ANY of the three candidate cutoffs
#   sits at a visible break, not just 50.
#
#   Fig 2 (formal RD plot): CCT (Calonico-Cattaneo-Titiunik)
#   binned-scatter + local polynomial fit on each side of the
#   50-mile cutoff specifically, same two years, using rdrobust's
#   rdplot (same package already used for the diff-in-disc check
#   in analyze_diffindisc.py -- this visualizes what that
#   regression is testing, on the same data).
#
# Same underlying data/construction as analyze_diffindisc.py
# (distance_by_year_rec_full.csv + the raw CDC VSRR pull), not a
# separately reconstructed sample.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdrobust import rdplot

BASE = os.path.dirname(os.path.abspath(__file__))
EL = os.path.dirname(BASE)
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

INK = '#2a2f2b'
INK_SOFT = '#5b655e'
GRID = '#e3e0d8'
DOT_2020 = '#8a8a8a'
DOT_2025 = '#c1502e'
LINE_2020 = '#5b5b5b'
LINE_2025 = '#8a331d'
CUTOFF_LINES = {50: '#2a5caa', 70: '#7a8c3f', 100: '#8a5a2a'}

dist = pd.read_csv(os.path.join(EL, 'data', 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
with open(os.path.join(EL, 'data', 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()
panel = dist.merge(od, on='fips5', how='left')


def year_frame(year):
    # Hawaii (island, thousands of miles from any legal-market county by
    # great-circle distance) is a standard map/analysis exclusion elsewhere
    # in this project (e.g. the CONUS map); its handful of counties would
    # otherwise dominate the x-axis scale and distort the polynomial fit.
    sub = panel[(panel['population'] > 0) & (panel['state_abbr'] != 'HI')].copy()
    sub['od_rate'] = sub[f'od_{year}'] / (sub['population'] / 1e5)
    sub['dist'] = sub[f'dist_{year}']
    return sub[sub['od_rate'].notna() & sub['dist'].notna()].copy()


# ---------------- Figure 1: scan across candidate cutoffs ----------------
BIN_WIDTH = 10
MAX_DIST = 250
bins = np.arange(0, MAX_DIST + BIN_WIDTH, BIN_WIDTH)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
for ax, year, dotcol, linecol in zip(axes, [2020, 2025], [DOT_2020, DOT_2025], [LINE_2020, LINE_2025]):
    d = year_frame(year)
    d = d[d['dist'] <= MAX_DIST]
    d['bin'] = pd.cut(d['dist'], bins=bins, include_lowest=True)
    g = d.groupby('bin', observed=True).agg(x=('dist', 'mean'), y=('od_rate', 'mean'), n=('od_rate', 'size')).dropna()
    ax.scatter(g['x'], g['y'], s=np.clip(g['n'] * 1.8, 10, 140), color=dotcol, alpha=0.85,
               edgecolor='white', linewidth=0.5, zorder=3)
    # light smoothed guide (rolling mean over bins, not a fitted model)
    roll = g['y'].rolling(3, center=True, min_periods=1)
    ax.plot(g['x'], roll.mean(), color=linecol, linewidth=1.4, alpha=0.8, zorder=2)
    for cutoff, col in CUTOFF_LINES.items():
        ax.axvline(cutoff, color=col, linestyle='--', linewidth=1.1, alpha=0.85, zorder=1)
        ax.text(cutoff, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, '', alpha=0)  # placeholder, ylim set after
    ax.set_xlabel('Distance to nearest legal-market county (miles)', fontsize=10, color=INK)
    ax.set_title(f'{year}', fontsize=12, loc='left', color=INK, fontweight='bold')
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SOFT, labelsize=9)

axes[0].set_ylabel('Overdose deaths per 100,000\n(binned county mean)', fontsize=10, color=INK)

# cutoff labels on the top axis only, placed after ylim is finalized
ymax = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(0, ymax)
    for cutoff, col in CUTOFF_LINES.items():
        ax.text(cutoff, ymax * 0.98, f'{cutoff}mi', color=col, fontsize=8.5, ha='center', va='top',
                fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.75))

fig.suptitle('Overdose Mortality vs. Distance to Nearest Legal-Market County: Scanning for a Break, 2020 vs. 2025',
             fontsize=12.5, color=INK, x=0.02, ha='left', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figI_distance_scan.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved figI_distance_scan.png')


# ---------------- Figure 2: formal CCT rdplot at 50mi ----------------
# Restricted to a local window around the cutoff (not the full national
# support, which runs out to 500+ miles for the farthest control
# counties) and a local-LINEAR fit (p=1, matching the order rdrobust
# itself uses for point estimation), rather than rdplot's p=4 default
# global-polynomial fit -- a quartic fit over the full support is
# dominated by a handful of very-far counties and produces wild swings
# that misrepresent what's happening near the cutoff.
CUTOFF = 50
WINDOW = 200  # miles on each side of the cutoff shown
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
for ax, year, dotcol, linecol in zip(axes, [2020, 2025], [DOT_2020, DOT_2025], [LINE_2020, LINE_2025]):
    d = year_frame(year)
    d = d[(d['dist'] >= CUTOFF - WINDOW) & (d['dist'] <= CUTOFF + WINDOW)]
    x = (d['dist'] - CUTOFF).to_numpy()
    y = d['od_rate'].to_numpy()
    out = rdplot(y=y, x=x, c=0, p=1, hide=True, kernel='triangular', binselect='esmv')
    bins_df = out.vars_bins
    poly_df = out.vars_poly
    ax.scatter(bins_df['rdplot_mean_x'], bins_df['rdplot_mean_y'],
               s=np.clip(bins_df['rdplot_N'] * 1.3, 8, 120), color=dotcol, alpha=0.85,
               edgecolor='white', linewidth=0.5, zorder=3)
    left = poly_df[poly_df['rdplot_x'] < 0]
    right = poly_df[poly_df['rdplot_x'] >= 0]
    ax.plot(left['rdplot_x'], left['rdplot_y'], color=linecol, linewidth=1.8, zorder=2)
    ax.plot(right['rdplot_x'], right['rdplot_y'], color=linecol, linewidth=1.8, zorder=2)
    ax.axvline(0, color=CUTOFF_LINES[50], linestyle='--', linewidth=1.2, zorder=1)
    ax.set_xlabel('Distance to nearest legal-market county $-$ 50 miles', fontsize=10, color=INK)
    ax.set_title(f'{year}', fontsize=12, loc='left', color=INK, fontweight='bold')
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK_SOFT, labelsize=9)

axes[0].set_ylabel('Overdose deaths per 100,000\n(CCT-optimal bin mean)', fontsize=10, color=INK)
fig.suptitle("Formal RD Plot at the 50-Mile Cutoff (Calonico–Cattaneo–Titiunik bins), 2020 vs. 2025",
             fontsize=12.5, color=INK, x=0.02, ha='left', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figJ_rdplot_50mi.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved figJ_rdplot_50mi.png')
