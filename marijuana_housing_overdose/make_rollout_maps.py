# ============================================================
# Multi-year small-multiple maps of the recreational-marijuana
# legal-market "rollout" and which never-legal-state counties
# fall within the 50-mile switcher/control radius as of each
# snapshot year. Kieran Healy-style static maps (muted palette,
# no chartjunk, direct-labeled legend, light neutral basemap).
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

# ── Healy-style palette (validated categorical order; muted, colorblind-aware) ──
INK = '#3f3f3f'
MUTED = '#8a8a8a'
GRID = '#e3e3e3'
LEGAL_ESTABLISHED = '#c7d9c0'   # muted sage, legal before this wave
LEGAL_NEW = '#4c9a6a'           # deeper green, newly legal this year
NEVERLEGAL_BASE = '#e8e4dc'     # warm light gray, never-legal state fill
POINT_FAR = '#b8b2a6'           # muted taupe, county NOT within radius
POINT_NEAR = '#c1502e'          # muted brick red, county WITHIN radius (Healy accent)

LEGAL_STATES = {
    'WA': 2014, 'CO': 2014, 'OR': 2015, 'AK': 2016, 'NV': 2017, 'MA': 2018,
    'CA': 2018, 'MI': 2019, 'IL': 2020, 'AZ': 2021, 'RI': 2022, 'VT': 2022,
    'MT': 2022, 'NJ': 2022, 'NM': 2022, 'NY': 2022, 'CT': 2023, 'MD': 2023,
    'MO': 2023, 'MN': 2024, 'OH': 2024,
}
NEVER_LEGAL = ['AL','AR','DC','DE','FL','GA','HI','IA','ID','IN','KS','KY','LA','ME',
               'MN','MS','NC','ND','NE','NH','OH','OK','PA','SC','SD','TN','TX','UT',
               'VA','WI','WV','WY']

states = gpd.read_file(os.path.join(DATA, 'us_states_shp', 'cb_2023_us_state_500k.shp'))
CONUS_EXCLUDE = {'AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP'}
states_conus = states[~states['STUSPS'].isin(CONUS_EXCLUDE)].to_crs(epsg=5070)  # Albers CONUS

dist = pd.read_csv(os.path.join(DATA, 'dist_by_year.csv'), dtype={'fips5': str})
panel = pd.read_stata(os.path.join(PARENT, 'marijuana_new', 'data', 'analysis_panel.dta'))
panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
meta = panel[panel['year'] == 2020][['fips5', 'lat', 'lon']].drop_duplicates('fips5')
dist = dist.merge(meta, on='fips5')
gdf_pts = gpd.GeoDataFrame(dist, geometry=gpd.points_from_xy(dist['lon'], dist['lat']), crs='EPSG:4326').to_crs(epsg=5070)

YEARS = [2020, 2021, 2022, 2023]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, year in zip(axes, YEARS):
    cutoff = 2019 if year == 2020 else year
    prior_cutoff = cutoff - 1 if year != 2020 else None
    established = [s for s, y in LEGAL_STATES.items() if y <= (prior_cutoff if year != 2020 else -1)]
    newly = [s for s, y in LEGAL_STATES.items() if y <= cutoff and s not in established]

    # basemap: never-legal states (warm light gray), established legal (sage), newly legal (deep green)
    base = states_conus.copy()
    def classify(usps):
        if usps in newly:
            return 'new'
        elif usps in established:
            return 'established'
        elif usps in NEVER_LEGAL:
            return 'never'
        else:
            return 'other'
    base['grp'] = base['STUSPS'].map(classify)
    color_map = {'new': LEGAL_NEW, 'established': LEGAL_ESTABLISHED, 'never': NEVERLEGAL_BASE, 'other': '#f5f5f2'}
    for grp, col in color_map.items():
        base[base['grp'] == grp].plot(ax=ax, color=col, edgecolor='white', linewidth=0.6)
    states_conus.boundary.plot(ax=ax, color='white', linewidth=0.6)

    # county points
    yr_pts = gdf_pts[gdf_pts['year'] == year]
    near = yr_pts[yr_pts['dist_mi'] < 50]
    far = yr_pts[yr_pts['dist_mi'] >= 50]
    far.plot(ax=ax, color=POINT_FAR, markersize=4, alpha=0.55, linewidth=0)
    near.plot(ax=ax, color=POINT_NEAR, markersize=11, alpha=0.9, linewidth=0)

    ax.set_xlim(states_conus.total_bounds[[0, 2]])
    ax.set_ylim(states_conus.total_bounds[[1, 3]])
    ax.set_axis_off()
    newly_label = ', '.join(sorted(newly))
    title = (f'{year}: legal baseline (pre-2020) -- {newly_label}' if year == 2020
              else f'{year}: newly legal this year -- {newly_label}')
    ax.set_title(title, fontsize=10, loc='left', color=INK, pad=6)

legend_elems = [
    Patch(facecolor=LEGAL_NEW, edgecolor='white', label='Newly legal this year'),
    Patch(facecolor=LEGAL_ESTABLISHED, edgecolor='white', label='Legal in a prior year'),
    Patch(facecolor=NEVERLEGAL_BASE, edgecolor='white', label='Never-legal state (sample)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_NEAR, markersize=8, label='County <50mi of legal border'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_FAR, markersize=6, label='County >=50mi of legal border'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Figure 1. The Legal-Market Rollout, 2020-2023: Never-Legal-State Counties Within 50 Miles of a Legal Border',
             fontsize=12, color=INK, y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(os.path.join(FIG, 'fig1_rollout_maps.png'), dpi=170, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig1_rollout_maps.png")
