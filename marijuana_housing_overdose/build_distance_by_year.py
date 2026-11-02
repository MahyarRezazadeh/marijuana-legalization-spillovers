# ============================================================
# Recomputes, for every never-legalizing-state county, its
# distance to the nearest recreational-marijuana-legal county
# as of each snapshot year 2020-2023 (not just the panel's
# precomputed 2020/2023 endpoints), plus which state achieves
# that minimum -- needed to build year-by-year rollout maps
# showing which border counties were newly "activated" by which
# state's legalization.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
DATA = os.path.join(BASE, 'data')

LEGAL_STATES = {  # recreational-legal effective year
    'WA': 2014, 'CO': 2014, 'OR': 2015, 'AK': 2016, 'NV': 2017, 'MA': 2018,
    'CA': 2018, 'MI': 2019, 'IL': 2020, 'AZ': 2021, 'RI': 2022, 'VT': 2022,
    'MT': 2022, 'NJ': 2022, 'NM': 2022, 'NY': 2022, 'CT': 2023, 'MD': 2023,
    'MO': 2023, 'MN': 2024, 'OH': 2024,
}
NEVER_LEGAL = ['AL','AR','DC','DE','FL','GA','HI','IA','ID','IN','KS','KY','LA','ME',
               'MN','MS','NC','ND','NE','NH','OH','OK','PA','SC','SD','TN','TX','UT',
               'VA','WI','WV','WY']
# NB: MN and OH appear in LEGAL_STATES with a 2024 date (after our panel window) and
# are ALSO in the historical NEVER_LEGAL list used by the companion project's 32-state
# never-legal panel (their law wasn't yet effective as of the 2023 snapshot), so they're
# correctly treated as never-legal throughout 2020-2023.

gaz = pd.read_csv(os.path.join(DATA, 'gaz', '2023_Gaz_counties_national.txt'), sep='\t')
gaz.columns = [c.strip() for c in gaz.columns]
gaz['GEOID'] = gaz['GEOID'].astype(str).str.zfill(5)
gaz['USPS'] = gaz['USPS'].str.strip()

panel = pd.read_stata(os.path.join(PARENT, 'marijuana_new', 'data', 'analysis_panel.dta'))
panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
meta = (panel[panel['year'] == 2020][['fips5', 'state_abbr', 'lat', 'lon',
                                        'dist_rec_2020_mi', 'dist_rec_2023_mi']]
        .drop_duplicates('fips5'))

def haversine_min(lat0, lon0, lats, lons):
    R = 3958.8  # miles
    p0, p1 = np.radians(lat0), np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi/2)**2 + np.cos(p0)*np.cos(p1)*np.sin(dlmb/2)**2
    d = 2*R*np.arcsin(np.sqrt(a))
    idx = np.argmin(d)
    return d[idx], idx

results = []
for year in [2020, 2021, 2022, 2023]:
    # The panel's 2020 baseline reflects status at the START of 2020 (excludes IL,
    # whose law took effect Jan 1 2020, confirmed by validation below); the 2023
    # (and by extension 2021/2022) snapshots reflect cumulative status through the
    # END of that year.
    cutoff = 2019 if year == 2020 else year
    legal_states_asof = [s for s, y in LEGAL_STATES.items() if y <= cutoff]
    legal_gaz = gaz[gaz['USPS'].isin(legal_states_asof)]
    llat, llon, lstate = legal_gaz['INTPTLAT'].to_numpy(), legal_gaz['INTPTLONG'].to_numpy(), legal_gaz['USPS'].to_numpy()
    for _, row in meta.iterrows():
        d, idx = haversine_min(row['lat'], row['lon'], llat, llon)
        results.append(dict(fips5=row['fips5'], year=year, dist_mi=d, nearest_state=lstate[idx]))
    print(f"year {year}: legal states = {sorted(legal_states_asof)}")

out = pd.DataFrame(results)
out.to_csv(os.path.join(DATA, 'dist_by_year.csv'), index=False)
print(f"\nSaved dist_by_year.csv ({len(out)} rows)")

# ── Validate against the panel's own precomputed 2020/2023 distances ──
check = out.pivot(index='fips5', columns='year', values='dist_mi').reset_index()
check = check.merge(meta[['fips5', 'dist_rec_2020_mi', 'dist_rec_2023_mi']], on='fips5')
check['diff_2020'] = check[2020] - check['dist_rec_2020_mi']
check['diff_2023'] = check[2023] - check['dist_rec_2023_mi']
print("\nValidation vs. panel's precomputed distances:")
print(check[['diff_2020', 'diff_2023']].describe())
print("\nWorst 5 mismatches (2020):")
print(check.reindex(check['diff_2020'].abs().sort_values(ascending=False).index).head(5)[
    ['fips5', 2020, 'dist_rec_2020_mi', 'diff_2020']])
