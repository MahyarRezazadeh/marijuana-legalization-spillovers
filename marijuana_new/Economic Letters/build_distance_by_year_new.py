import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MARIJUANA_NEW = os.path.dirname(BASE)
ROOT = os.path.dirname(MARIJUANA_NEW)
DATA = os.path.join(BASE, 'data')
HOUSING_DATA = os.path.join(ROOT, 'marijuana_housing_overdose', 'data')

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['effective_date', 'retail_date'])

gaz = pd.read_csv(os.path.join(HOUSING_DATA, 'gaz', '2023_Gaz_counties_national.txt'), sep='\t')
gaz.columns = [c.strip() for c in gaz.columns]
gaz['GEOID'] = gaz['GEOID'].astype(str).str.zfill(5)
gaz['USPS'] = gaz['USPS'].str.strip()

panel = pd.read_stata(os.path.join(MARIJUANA_NEW, 'data', 'analysis_panel.dta'))
panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
meta = panel[panel['year'] == 2020][['fips5', 'state_abbr', 'lat', 'lon']].drop_duplicates('fips5')
meta = meta[meta['state_abbr'] != 'ME'].copy()

def haversine_min(lat0, lon0, lats, lons):
    R = 3958.8
    p0, p1 = np.radians(lat0), np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi/2)**2 + np.cos(p0)*np.cos(p1)*np.sin(dlmb/2)**2
    d = 2*R*np.arcsin(np.sqrt(a))
    idx = np.argmin(d)
    return d[idx], idx

YEARS = [2020, 2021, 2022, 2023]
results = []
for year in YEARS:
    cutoff = pd.Timestamp(f'{year}-12-31')
    legal_states_asof = sorted(legal[legal['retail_date'] <= cutoff]['state_abbr'])
    legal_gaz = gaz[gaz['USPS'].isin(legal_states_asof)]
    llat, llon, lstate = legal_gaz['INTPTLAT'].to_numpy(), legal_gaz['INTPTLONG'].to_numpy(), legal_gaz['USPS'].to_numpy()
    for _, row in meta.iterrows():
        d, idx = haversine_min(row['lat'], row['lon'], llat, llon)
        results.append(dict(fips5=row['fips5'], year=year, dist_mi=d, nearest_state=lstate[idx]))
    print(f"year {year}: legal-market states = {legal_states_asof}")

out = pd.DataFrame(results)
out.to_csv(os.path.join(DATA, 'dist_by_year_new.csv'), index=False)
print(f"\nSaved dist_by_year_new.csv ({len(out)} rows)")
