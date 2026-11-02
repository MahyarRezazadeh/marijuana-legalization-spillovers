# ============================================================
# Simple single-year (2023) treatment/control map for the slide
# deck: never-legal-state counties colored by whether their
# distance to the nearest legal-market county is below the
# 50-mile threshold (treatment, red) or not (control, blue), as
# of the December 2023 snapshot. Legal-market states themselves
# are shaded for context but contribute no dots (outside the
# 31-state/2,331-county estimation sample).
# ============================================================
import os
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.abspath(__file__))
EL = os.path.dirname(BASE)
ROOT = os.path.dirname(os.path.dirname(EL))
HOUSING = os.path.join(ROOT, 'marijuana_housing_overdose')
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

INK = '#2a2f2b'
NEVERLEGAL_FILL = '#e9e6dd'   # never-legal state (sample universe)
LEGAL_FILL = '#cfd9cc'        # legal-market state (outside sample)
CONTROL_BLUE = '#2A5CAA'
TREATMENT_RED = '#C0392B'

dist = pd.read_csv(os.path.join(EL, 'data', 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
NEVER_LEGAL_STATES = sorted(dist['state_abbr'].unique().tolist())

states = gpd.read_file(os.path.join(HOUSING, 'data', 'us_states_shp', 'cb_2023_us_state_500k.shp'))
CONUS_EXCLUDE = {'AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP'}
states_conus = states[~states['STUSPS'].isin(CONUS_EXCLUDE)].to_crs(epsg=5070)
states_conus['grp'] = states_conus['STUSPS'].apply(lambda s: 'never_legal' if s in NEVER_LEGAL_STATES else 'legal')

pts = dist.copy()
pts['status'] = pts['dist_2023'].apply(lambda d: 'Treatment' if d < 50 else 'Control')
gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts['lon'], pts['lat']), crs='EPSG:4326').to_crs(epsg=5070)
gdf_conus = gdf[~gdf['state_abbr'].isin(CONUS_EXCLUDE)]

n_treat = int((gdf_conus['status'] == 'Treatment').sum())
n_control = int((gdf_conus['status'] == 'Control').sum())
print(f"2023 snapshot (contiguous US): {n_treat} treatment counties, {n_control} control counties")

fig, ax = plt.subplots(figsize=(11, 6.8))

for grp, col in [('legal', LEGAL_FILL), ('never_legal', NEVERLEGAL_FILL)]:
    states_conus[states_conus['grp'] == grp].plot(ax=ax, color=col, edgecolor='white', linewidth=0.7)

control = gdf_conus[gdf_conus['status'] == 'Control']
treat = gdf_conus[gdf_conus['status'] == 'Treatment']
control.plot(ax=ax, color=CONTROL_BLUE, markersize=9, alpha=0.75, linewidth=0, zorder=3)
treat.plot(ax=ax, color=TREATMENT_RED, markersize=16, alpha=0.9, linewidth=0, zorder=4)

ax.set_xlim(states_conus.total_bounds[[0, 2]])
ax.set_ylim(states_conus.total_bounds[[1, 3]])
ax.set_axis_off()
ax.set_title('Treatment and Control Counties, December 2023 Snapshot', fontsize=14, loc='left', color=INK, pad=10, fontweight='bold')

legend_elems = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=TREATMENT_RED, markersize=9,
           label=f'Treatment: <50 miles of a legal-market county  (n={n_treat})'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=CONTROL_BLUE, markersize=8,
           label=f'Control: ≥50 miles of a legal-market county  (n={n_control})'),
    Patch(facecolor=LEGAL_FILL, edgecolor='white', label='Legal-market state (outside estimation sample)'),
    Patch(facecolor=NEVERLEGAL_FILL, edgecolor='white', label='Never-legal state (31-state sample universe)'),
]
ax.legend(handles=legend_elems, loc='lower right', frameon=False, fontsize=9.5, labelcolor=INK)

plt.tight_layout()
outpath = os.path.join(FIG, 'map_treatment_control_2023.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved {outpath}")
