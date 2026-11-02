# ============================================================
# Figures for the new stacked multi-period design (all state
# dates 2020-2025, recreational primary + medical robustness).
# Kieran Healy style, consistent with the rest of the project.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.abspath(__file__))
MARIJUANA_NEW = os.path.dirname(BASE)
ROOT = os.path.dirname(MARIJUANA_NEW)
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')
HOUSING_DATA = os.path.join(ROOT, 'marijuana_housing_overdose', 'data')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#c1502e'; ACCENT2 = '#4c9a6a'; GRAY = '#8a8a8a'
LEGAL_NEW = '#4c9a6a'; LEGAL_EST = '#c7d9c0'; NEVERLEGAL = '#e8e4dc'
POINT_FAR = '#b8b2a6'; POINT_NEAR = '#c1502e'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['effective_date', 'retail_date'])
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# ============================================================
# Fig 1: rollout maps, 2020-2025 (6 panels)
# ============================================================
states = gpd.read_file(os.path.join(HOUSING_DATA, 'us_states_shp', 'cb_2023_us_state_500k.shp'))
CONUS_EXCLUDE = {'AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP'}
states_conus = states[~states['STUSPS'].isin(CONUS_EXCLUDE)].to_crs(epsg=5070)

dist_wide = pd.read_csv(os.path.join(DATA, 'distance_by_year_rec.csv'), dtype={'fips5': str})
gdf_pts = gpd.GeoDataFrame(dist_wide, geometry=gpd.points_from_xy(dist_wide['lon'], dist_wide['lat']), crs='EPSG:4326').to_crs(epsg=5070)

panel_full = pd.read_stata(os.path.join(MARIJUANA_NEW, 'data', 'analysis_panel.dta'))
COMPANION_NEVERLEGAL = sorted(set(panel_full[panel_full['never_legal'] == 1]['state_abbr'].unique()) - {'ME'})

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for ax, year in zip(axes, YEARS):
    cutoff = pd.Timestamp(f'{year}-12-31')
    established = sorted(legal[legal['retail_date'] <= pd.Timestamp(f'{year-1}-12-31')]['state_abbr']) if year > 2020 else []
    newly = sorted(set(legal[legal['retail_date'] <= cutoff]['state_abbr']) - set(established))

    base = states_conus.copy()
    def classify(usps):
        if usps in newly: return 'new'
        elif usps in established: return 'established'
        elif usps in COMPANION_NEVERLEGAL: return 'never'
        else: return 'other'
    base['grp'] = base['STUSPS'].map(classify)
    color_map = {'new': LEGAL_NEW, 'established': LEGAL_EST, 'never': NEVERLEGAL, 'other': '#f5f5f2'}
    for grp, col in color_map.items():
        base[base['grp'] == grp].plot(ax=ax, color=col, edgecolor='white', linewidth=0.6)
    states_conus.boundary.plot(ax=ax, color='white', linewidth=0.6)

    dcol = f'dist_{year}'
    near = gdf_pts[gdf_pts[dcol] < 50]
    far = gdf_pts[gdf_pts[dcol] >= 50]
    far.plot(ax=ax, color=POINT_FAR, markersize=3, alpha=0.5, linewidth=0)
    near.plot(ax=ax, color=POINT_NEAR, markersize=9, alpha=0.9, linewidth=0)

    ax.set_xlim(states_conus.total_bounds[[0, 2]]); ax.set_ylim(states_conus.total_bounds[[1, 3]])
    ax.set_axis_off()
    newly_label = ', '.join(newly) if newly else 'none'
    title = f'{year}: baseline -- {newly_label}' if year == 2020 else f'{year}: newly retail-legal -- {newly_label}'
    ax.set_title(title, fontsize=9, loc='left', color=INK, pad=5)

legend_elems = [
    Patch(facecolor=LEGAL_NEW, edgecolor='white', label='Newly retail-legal this year'),
    Patch(facecolor=LEGAL_EST, edgecolor='white', label='Retail-legal in a prior year'),
    Patch(facecolor=NEVERLEGAL, edgecolor='white', label='Never-legal state (sample)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_NEAR, markersize=8, label='County <50mi of legal-market border'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_FAR, markersize=6, label='County >=50mi of legal-market border'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Figure 1. The Legal-Market Rollout (by first retail sales date), 2020-2025', fontsize=12, x=0.02, ha='left', y=0.99, color=INK)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(os.path.join(FIG, 'fig1_rollout_maps_multiperiod.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig1_rollout_maps_multiperiod.png")

# ============================================================
# Fig 2: coefficient plot -- recreational (primary) vs medical (robustness)
# ============================================================
with open(os.path.join(DATA, 'multiperiod_results_rec.json')) as f: rec = json.load(f)
with open(os.path.join(DATA, 'multiperiod_results_med.json')) as f: med = json.load(f)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
labels = ['Recreational: switcher\n(binary, <50mi)', 'Recreational: distance\n(per 100mi, for scale)',
          'Medical (placebo): switcher\n(binary, <50mi)', 'Medical (placebo): distance\n(per 100mi, for scale)']
specs = [(rec['switcher'], 1), (rec['dist_mile'], 100), (med['switcher'], 1), (med['dist_mile'], 100)]
coefs, los, his, cols = [], [], [], []
for spec, scale in specs:
    b, se, p = spec['coef'] * scale, spec['se'] * scale, spec['p']
    coefs.append(b); los.append(b - 1.96*se); his.append(b + 1.96*se)
    cols.append(ACCENT if p < 0.05 else GRAY)
y = np.arange(len(labels))
for i in range(len(labels)):
    ax.errorbar(coefs[i], y[i], xerr=[[coefs[i]-los[i]], [his[i]-coefs[i]]], fmt='o', color=cols[i], capsize=5, ms=10, elinewidth=2)
ax.axvline(0, color='#bbb', lw=1, linestyle='--')
ax.axhline(1.5, color='#ddd', lw=0.8, linestyle=':')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.8); ax.invert_yaxis()
healy(ax, title='Figure 2. Effect on Drug-Overdose Mortality, Pooled Stacked-Transition Panel (95% CI)',
      xlabel='Change in overdose deaths per 100,000')
plt.tight_layout(rect=[0, 0.18, 1, 0.94])
fig.text(0.02, 0.02, 'SE clustered by state, transition fixed effects. Colored marker = significant at 5%. The distance rows are signed\n'
                      'as the effect of an INCREASE in distance, so a positive distance coefficient and a negative switcher coefficient point\n'
                      'the same direction (getting closer to a border reduces overdose deaths). Medical row uses medical-access dates in\n'
                      'place of recreational-retail dates as a placebo-style check.',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig2_overdose_coefplot_multiperiod.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig2_overdose_coefplot_multiperiod.png")

# ============================================================
# Fig 3: event study -- per-transition switcher coefficients (rec)
# ============================================================
pt = pd.read_csv(os.path.join(BASE, 'tables', 'event_study_rec.csv'))
fig, ax = plt.subplots(figsize=(7, 4))
xlabels = pt['transition'].tolist()
xpos = np.arange(len(xlabels))
cols = [ACCENT if p < 0.05 else GRAY for p in pt['p']]
for i, row in pt.iterrows():
    lo, hi = row['coef'] - 1.96*row['se'], row['coef'] + 1.96*row['se']
    ax.errorbar(xpos[i], row['coef'], yerr=[[row['coef']-lo], [hi-row['coef']]], fmt='o', color=cols[i], capsize=5, ms=10, elinewidth=2)
ax.axhline(0, color='#bbb', lw=1, linestyle='--')
ax.axhline(rec['switcher']['coef'], color=INK, lw=1.2, linestyle=':', alpha=0.6)
ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=9)
healy(ax, title='Figure 3. The Switcher Effect Is Not Driven by a Single Period',
      xlabel='Transition (year-over-year)', ylabel='Switcher coefficient (per 100,000)')
plt.tight_layout(rect=[0, 0.1, 1, 1])
fig.text(0.02, 0.02, 'Dotted horizontal line = pooled estimate. Transitions with too few switch events after suppression are omitted.',
          fontsize=7.5, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig3_eventstudy_multiperiod.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig3_eventstudy_multiperiod.png")

# ============================================================
# Fig 4: continuous scatter, pooled across all transitions (rec)
# ============================================================
s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'))
ok = s['d_dist_mi'].notna() & s['d_od'].notna()
x, y = s.loc[ok, 'd_dist_mi'], s.loc[ok, 'd_od']
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.scatter(x, y, s=10, alpha=0.28, color=GRAY, linewidths=0)
z = np.polyfit(x, y, 1)
xr = np.linspace(x.quantile(0.02), x.quantile(0.98), 50)
ax.plot(xr, np.polyval(z, xr), color=ACCENT, lw=2)
ax.axvline(0, color='#ddd', lw=0.8); ax.axhline(0, color='#ddd', lw=0.8)
healy(ax, title='Figure 4. The Continuous-Distance Regression, Pooled Across All Transitions',
      xlabel='Change in distance to nearest legal-market border (miles)', ylabel='Change in overdose rate per 100,000')
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.text(0.02, 0.01, 'Negative x = county became closer to a newly retail-legal border. Pools all 5 year-over-year transitions, 2020-2025.',
          fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig4_overdose_scatter_multiperiod.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig4_overdose_scatter_multiperiod.png")

# ============================================================
# Fig A: dose-response, pooled multi-period
# ============================================================
grid = pd.read_csv(os.path.join(DATA, 'doseresponse_grid_multiperiod.csv'))
with open(os.path.join(DATA, 'doseresponse_multiperiod.json')) as f: dr = json.load(f)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
ax.scatter(x, y, s=8, alpha=0.22, color=GRAY, linewidths=0)
ax.plot(grid['d_dist_mi'], grid['pred_quad'], color=ACCENT, lw=2, label='Quadratic fit')
ax.fill_between(grid['d_dist_mi'], grid['quad_lo'], grid['quad_hi'], color=ACCENT, alpha=0.15, linewidth=0)
ax.plot(grid['d_dist_mi'], grid['pred_lin'], color=INK, lw=1.4, linestyle='--', label='Linear fit')
ax.legend(fontsize=8, frameon=False)
ax.axvline(0, color='#ddd', lw=0.8); ax.axhline(0, color='#ddd', lw=0.8)
healy(ax, title='(a) Dose-response: quadratic vs. linear', xlabel='Change in distance (miles)', ylabel='Change in overdose rate per 100,000')

ax = axes[1]
me_lin = np.full(len(grid), dr['linear']['coef'])
me_quad = dr['linear']['coef'] + 2 * dr['quad_term']['coef'] * grid['d_dist_mi']
ax.plot(grid['d_dist_mi'], me_lin, color=INK, lw=1.6, linestyle='--', label='Linear spec (constant)')
ax.plot(grid['d_dist_mi'], me_quad, color=ACCENT, lw=2, label='Quadratic spec (marginal effect)')
ax.axhline(0, color='#ddd', lw=0.8); ax.axvline(0, color='#ddd', lw=0.8)
ax.legend(fontsize=8, frameon=False)
healy(ax, title='(b) Implied marginal effect of +1 mile', xlabel='Change in distance (miles)', ylabel='Marginal effect on overdose rate')

fig.suptitle('Figure A. Dose-Response Is Nonlinear in the Multi-Period Panel', fontsize=11.5, x=0.02, ha='left', y=0.99, color=INK)
plt.tight_layout(rect=[0, 0.12, 1, 0.90])
fig.text(0.02, 0.02, f"Quadratic term F-test p={dr['f_quad']['p']:.4f}; quad+cubic joint F-test p={dr['f_quadcubic']['p']:.4f} "
                     f"-- curvature is statistically significant in this richer panel (unlike the earlier two-snapshot design).",
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figA_doseresponse_multiperiod.png'), dpi=170, facecolor='white')
plt.close()
print("Saved figA_doseresponse_multiperiod.png")
