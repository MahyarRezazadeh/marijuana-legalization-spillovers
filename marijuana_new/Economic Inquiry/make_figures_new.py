# ============================================================
# Rebuilds all main-text figures with the corrected legalization
# timing data (retail-sales-date cutoffs). Kieran Healy style,
# consistent with the rest of the project.
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
import statsmodels.api as sm

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
LEGAL_MAP = dict(zip(legal['state_abbr'], legal['retail_date']))
NEVER_OR_NOT_YET = set(legal[legal['retail_date'].isna() | (legal['retail_date'] > pd.Timestamp('2023-12-31'))]['state_abbr'])

# ============================================================
# Fig 1: rollout maps
# ============================================================
states = gpd.read_file(os.path.join(HOUSING_DATA, 'us_states_shp', 'cb_2023_us_state_500k.shp'))
CONUS_EXCLUDE = {'AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP'}
states_conus = states[~states['STUSPS'].isin(CONUS_EXCLUDE)].to_crs(epsg=5070)

dist = pd.read_csv(os.path.join(DATA, 'dist_by_year_new.csv'), dtype={'fips5': str})
panel_full = pd.read_stata(os.path.join(MARIJUANA_NEW, 'data', 'analysis_panel.dta'))
panel_full['fips5'] = panel_full['fips5'].astype(str).str.zfill(5)
metaxy = panel_full[panel_full['year'] == 2020][['fips5', 'lat', 'lon']].drop_duplicates('fips5')
dist = dist.merge(metaxy, on='fips5')
gdf_pts = gpd.GeoDataFrame(dist, geometry=gpd.points_from_xy(dist['lon'], dist['lat']), crs='EPSG:4326').to_crs(epsg=5070)

NEVER_LEGAL_31 = sorted(set(panel_full['state_abbr'].unique()) - {'ME'} - set(legal['state_abbr']))
# (Actually never-legal-31 = companion 32-state list minus ME; states in `legal` besides ME are excluded below via NEVER_LEGAL set)
COMPANION_NEVERLEGAL = sorted(set(panel_full[panel_full['never_legal'] == 1]['state_abbr'].unique()) - {'ME'})

YEARS = [2020, 2021, 2022, 2023]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
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

    yr_pts = gdf_pts[gdf_pts['year'] == year]
    near = yr_pts[yr_pts['dist_mi'] < 50]
    far = yr_pts[yr_pts['dist_mi'] >= 50]
    far.plot(ax=ax, color=POINT_FAR, markersize=4, alpha=0.55, linewidth=0)
    near.plot(ax=ax, color=POINT_NEAR, markersize=11, alpha=0.9, linewidth=0)

    ax.set_xlim(states_conus.total_bounds[[0, 2]]); ax.set_ylim(states_conus.total_bounds[[1, 3]])
    ax.set_axis_off()
    newly_label = ', '.join(newly) if newly else 'none'
    title = (f'{year}: legal-market baseline (retail open pre-2020) -- {newly_label}'
             if year == 2020 else f'{year}: newly retail-legal this year -- {newly_label}')
    ax.set_title(title, fontsize=9.5, loc='left', color=INK, pad=6)

legend_elems = [
    Patch(facecolor=LEGAL_NEW, edgecolor='white', label='Newly retail-legal this year'),
    Patch(facecolor=LEGAL_EST, edgecolor='white', label='Retail-legal in a prior year'),
    Patch(facecolor=NEVERLEGAL, edgecolor='white', label='Never-legal state (sample)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_NEAR, markersize=8, label='County <50mi of legal-market border'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_FAR, markersize=6, label='County >=50mi of legal-market border'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Figure 1. The Legal-Market Rollout (by first retail sales date), 2020-2023', fontsize=12, x=0.02, ha='left', y=0.98, color=INK)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(os.path.join(FIG, 'fig1_rollout_maps_new.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig1_rollout_maps_new.png")

# ============================================================
# Fig 2: coefficient plot
# ============================================================
with open(os.path.join(DATA, 'overdose_results_new.json')) as f:
    res = json.load(f)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
labels = ['Switcher\n(binary, <50mi)', 'Distance\n(per 100mi, for scale)', 'Log-distance\n(robustness)']
keys = ['switcher', 'dist_mile', 'log_dist']
scale = {'switcher': 1, 'dist_mile': 100, 'log_dist': 1}  # dist_mile plotted per-100mi for visual comparability only; table/text stay per-mile
coefs, los, his, cols = [], [], [], []
for k in keys:
    b, se, p, n = res[k]
    b, se = b * scale[k], se * scale[k]
    coefs.append(b); los.append(b - 1.96*se); his.append(b + 1.96*se)
    cols.append(ACCENT if p < 0.05 else GRAY)
y = np.arange(len(labels))
for i in range(len(labels)):
    ax.errorbar(coefs[i], y[i], xerr=[[coefs[i]-los[i]], [his[i]-coefs[i]]], fmt='o', color=cols[i], capsize=5, ms=10, elinewidth=2)
ax.axvline(0, color='#bbb', lw=1, linestyle='--')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
healy(ax, title='Figure 2. Effect on Drug-Overdose Mortality (95% CI)', xlabel='Change in overdose deaths per 100,000')
plt.tight_layout(rect=[0, 0.1, 1, 1])
fig.text(0.02, 0.02, 'SE clustered by state. Colored marker = significant at 5%.', fontsize=7.5, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig2_overdose_coefplot_new.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig2_overdose_coefplot_new.png")

# ============================================================
# Fig 3: switcher vs control trend
# ============================================================
panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel_new.csv'))
avail = panel[panel['d_od'].notna()]
tr = avail.groupby('switcher')[['od_rate20', 'od_rate23']].mean()
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for swv, col, lab in [(0, GRAY, 'Control'), (1, ACCENT, 'Switcher')]:
    ax.plot([2020, 2023], [tr.loc[swv, 'od_rate20'], tr.loc[swv, 'od_rate23']], color=col, marker='o', ms=8, lw=2.2)
    ax.annotate(lab, (2023, tr.loc[swv, 'od_rate23']), xytext=(6, 0), textcoords='offset points', fontsize=9, color=col, va='center')
ax.set_xticks([2020, 2023]); ax.set_xlim(2019.7, 2023.6)
healy(ax, title='Figure 3. Overdose Death Rate: Switchers vs. Controls', ylabel='Deaths per 100,000 (12-month-ending)')
plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.text(0.02, 0.015, 'CDC VSRR data, non-suppressed counties only.', fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig3_overdose_trends_new.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig3_overdose_trends_new.png")

# ============================================================
# Fig 4: continuous scatter
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4.5))
x, y = panel['d_dist_mi'], panel['d_od']
ok = x.notna() & y.notna()
ax.scatter(x[ok], y[ok], s=12, alpha=0.35, color=GRAY, linewidths=0)
z = np.polyfit(x[ok], y[ok], 1)
xr = np.linspace(x[ok].quantile(0.02), x[ok].quantile(0.98), 50)
ax.plot(xr, np.polyval(z, xr), color=ACCENT, lw=2)
ax.axvline(0, color='#ddd', lw=0.8); ax.axhline(0, color='#ddd', lw=0.8)
healy(ax, title='Figure 4. The Continuous-Distance Regression', xlabel='Change in distance to nearest legal-market border (miles)', ylabel='Change in overdose rate per 100,000')
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.text(0.02, 0.01, 'Negative x = county became closer to a newly retail-legal border.', fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig4_overdose_scatter_new.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig4_overdose_scatter_new.png")
