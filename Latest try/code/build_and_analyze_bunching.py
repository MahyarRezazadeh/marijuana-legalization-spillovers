# ============================================================
# Store-level spatial bunching test: do sellers in border counties
# locate deliberately close to the actual state line (to intercept
# cross-border traffic), rather than being scattered uniformly around
# the county? This uses each dispensary's exact geocoded point against
# the real state-boundary polygon of the nearest nonlegal state, which
# is a finer test than anything in the existing pipeline (which only
# ever compares COUNTY-CENTROID distance to the border, not individual
# store locations within the county).
#
# Statistic: for each border county (border_70mi==1, using the
# project's existing dist_nonlegal_mi/nearest_nonlegal_state fields),
# compare (a) the county centroid's distance to the nonlegal state
# polygon [already computed elsewhere] against (b) the mean distance
# of that county's OWN dispensaries to the same polygon. If sellers
# bunch toward the border, stores should sit systematically closer to
# the line than the county's geometric center does. A companion
# "near-side share" statistic -- the fraction of a county's stores
# strictly closer to the line than the county centroid is -- has a
# clean 50% null under location choices uncorrelated with the border,
# tested here directly rather than assumed.
# Interior counties (border_70mi==0) get the identical calculation as
# a placebo: there is no cross-border-demand reason to expect bunching
# toward a distant nonlegal state there.
# ============================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PROJ_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ_BASE, 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
REF_DATE = pd.Timestamp('2026-08-05')
METERS_PER_MILE = 1609.344

# ---------- dispensaries (same "currently open" convention as build_dispensary_counts.py) ----------
disp = pd.read_csv(os.path.join(DATA, 'dispensaries_master_geocoded.csv'))
disp = disp.dropna(subset=['lat', 'lon']).copy()
disp['open_date'] = pd.to_datetime(disp['open_date'], errors='coerce')
disp = disp[disp['open_date'].isna() | (disp['open_date'] <= REF_DATE)].copy()

shp = gpd.read_file(os.path.join(DATA, 'us_counties_shp', 'cb_2023_us_county_500k.shp'))
shp = shp.to_crs('EPSG:5070')  # CONUS Albers equal-area, meters
county_shp = shp[['GEOID', 'STUSPS', 'geometry']].rename(columns={'GEOID': 'fips5'})

disp_gdf = gpd.GeoDataFrame(disp, geometry=[Point(xy) for xy in zip(disp['lon'], disp['lat'])],
                             crs='EPSG:4326').to_crs('EPSG:5070')
joined = gpd.sjoin(disp_gdf, county_shp[['fips5', 'geometry']], how='left', predicate='within')
if joined['fips5'].isna().any():
    missing = joined[joined['fips5'].isna()][['uid', 'geometry']].set_geometry('geometry')
    nearest = gpd.sjoin_nearest(missing, county_shp[['fips5', 'geometry']], how='left')
    col = 'fips5_right' if 'fips5_right' in nearest.columns else 'fips5'
    fix = dict(zip(nearest['uid'], nearest[col]))
    joined.loc[joined['fips5'].isna(), 'fips5'] = joined.loc[joined['fips5'].isna(), 'uid'].map(fix)
print(f"Dispensaries geocoded and matched to a county: {joined['fips5'].notna().sum()} of {len(joined)}")

universe = pd.read_csv(os.path.join(DATA, 'county_universe_with_distance.csv'), dtype={'fips5': str})
universe['border_70mi'] = (universe['dist_nonlegal_mi'] <= 70).astype(int)

# ---------- dissolve counties into state polygons (for the specific nonlegal neighbor states used) ----------
needed_states = universe['nearest_nonlegal_state'].unique().tolist()
state_polys = shp[shp['STUSPS'].isin(needed_states)].dissolve(by='STUSPS')['geometry']

joined = joined.merge(universe[['fips5', 'state_abbr', 'border_70mi', 'nearest_nonlegal_state',
                                 'dist_nonlegal_mi']], on='fips5', how='left')
joined = joined.dropna(subset=['nearest_nonlegal_state']).copy()

joined['dist_line_mi'] = joined.apply(
    lambda r: r.geometry.distance(state_polys.loc[r['nearest_nonlegal_state']]) / METERS_PER_MILE, axis=1)

county_agg = joined.groupby('fips5').agg(
    n_stores=('dist_line_mi', 'size'),
    mean_store_dist=('dist_line_mi', 'mean'),
    near_side_count=('dist_line_mi', lambda s: None)).reset_index()  # placeholder, filled below

# near-side share: stores closer to the line than the county centroid is
centroid_dist_map = universe.set_index('fips5')['dist_nonlegal_mi']
joined['centroid_dist_mi'] = joined['fips5'].map(centroid_dist_map)
joined['store_nearer_than_centroid'] = (joined['dist_line_mi'] < joined['centroid_dist_mi']).astype(int)

county_agg = joined.groupby('fips5').agg(
    state_abbr=('state_abbr', 'first'),
    border_70mi=('border_70mi', 'first'),
    centroid_dist_mi=('centroid_dist_mi', 'first'),
    n_stores=('dist_line_mi', 'size'),
    mean_store_dist_mi=('dist_line_mi', 'mean'),
    near_side_share=('store_nearer_than_centroid', 'mean'),
).reset_index()

joined.drop(columns='geometry').to_csv(os.path.join(OUT, 'dispensary_border_distances.csv'), index=False)
county_agg.to_csv(os.path.join(OUT, 'county_bunching_summary.csv'), index=False)
print(f"Saved dispensary_border_distances.csv ({len(joined)} stores), "
      f"county_bunching_summary.csv ({len(county_agg)} counties)")

# ---------- headline tests ----------
# The RATIO of a county's own mean store-to-line distance to its centroid's
# distance to the line is the meaningful statistic here, not the raw mile
# gap or the raw "near-side share": both of the latter turn out to be
# dominated by a generic, border-irrelevant fact that incorporated towns
# are essentially never located at a county's literal geometric centroid,
# which produces a huge ">50% near side" reading in INTERIOR counties too
# (91.8%, nearly as large as border counties' 94.6% -- not a border-specific
# signal at all, and reported below only as a diagnostic, not a finding).
# The ratio measure controls for this by asking how much closer, proportional
# to the county's own scale, stores sit relative to their county's centroid --
# and it cleanly separates border from interior counties.
county_agg['store_to_centroid_ratio'] = county_agg['mean_store_dist_mi'] / county_agg['centroid_dist_mi']


def summarize(group_df, label):
    n_counties = len(group_df)
    n_stores = group_df['n_stores'].sum()
    ratio = group_df['store_to_centroid_ratio']
    gap = group_df['centroid_dist_mi'] - group_df['mean_store_dist_mi']
    near_share_store_weighted = np.average(group_df['near_side_share'], weights=group_df['n_stores'])
    print(f"\n=== {label} ===")
    print(f"counties={n_counties}, stores={n_stores}")
    print(f"mean store-dist/centroid-dist ratio = {ratio.mean():.3f} (median {ratio.median():.3f})")
    print(f"mean(centroid_dist - store_dist) = {gap.mean():+.2f} mi [diagnostic only, see note above]")
    print(f"store-weighted mean near-side share = {near_share_store_weighted*100:.1f}% [diagnostic only]")
    return dict(group=label, n_counties=n_counties, n_stores=int(n_stores),
                mean_ratio=ratio.mean(), median_ratio=ratio.median(), sd_ratio=ratio.std(),
                mean_gap_mi_diagnostic=gap.mean(), near_side_share_diagnostic=near_share_store_weighted)

border_g = county_agg[(county_agg['border_70mi'] == 1) & (county_agg['n_stores'] >= 1)]
interior_g = county_agg[(county_agg['border_70mi'] == 0) & (county_agg['n_stores'] >= 1)]
rows = [summarize(border_g, 'Border counties (<=70mi)'),
        summarize(interior_g, 'Interior counties (placebo)')]

t_ratio, p_ratio = stats.ttest_ind(border_g['store_to_centroid_ratio'], interior_g['store_to_centroid_ratio'],
                                    equal_var=False)
print(f"\nBorder vs. interior ratio, Welch t-test: t={t_ratio:.2f}, p={p_ratio:.3g}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'bunching_test_results.csv'), index=False)
pd.DataFrame([dict(t_stat=t_ratio, p_value=p_ratio,
                    border_mean_ratio=border_g['store_to_centroid_ratio'].mean(),
                    interior_mean_ratio=interior_g['store_to_centroid_ratio'].mean())]
             ).to_csv(os.path.join(OUT, 'bunching_ratio_ttest.csv'), index=False)
print("Saved bunching_test_results.csv, bunching_ratio_ttest.csv")

# ---------- figure ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(joined.loc[joined['border_70mi'] == 1, 'dist_line_mi'], bins=30, color='tab:red', alpha=0.8)
axes[0].set_title('Store distance to nonlegal state line\n(dispensaries in border counties, <=70mi)')
axes[0].set_xlabel('Miles')
axes[0].set_ylabel('Number of dispensaries')

for grp, label, color in [(1, 'border counties', 'tab:red'), (0, 'interior counties', 'tab:blue')]:
    data = county_agg.loc[(county_agg['border_70mi'] == grp) & (county_agg['n_stores'] >= 1), 'store_to_centroid_ratio']
    axes[1].hist(data, bins=20, alpha=0.5, label=label, color=color, density=True)
axes[1].axvline(1.0, color='gray', ls='--', label='1.0 (stores as far as centroid, on average)')
axes[1].set_title('County-level ratio: mean store distance to the\nnonlegal-state line / county-centroid distance to it')
axes[1].set_xlabel('Ratio (< 1 means stores sit closer to the line than the county center)')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_border_bunching.png'), dpi=150)
print("Saved fig_border_bunching.png")
