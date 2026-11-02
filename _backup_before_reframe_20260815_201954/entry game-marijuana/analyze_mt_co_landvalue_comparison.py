# ============================================================
# Does farmland value near a nonlegal border behave differently in
# Montana (uncapped entry) than in Colorado (capped entry)? Reuses the
# exact same national NASS land-value panel already built for the
# main (pooled, 23-state) land-value border analysis
# (analyze_land_value_border.py) -- no new data pull needed -- just
# cut to MT-only and CO-only subsamples.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
THRESH = 70

geo = pd.read_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'), dtype={'fips5': str})

nass_dir = os.path.join(BASE, '..', 'marijuana_new', 'health outcomes exploration', 'data')
nass = pd.read_csv(os.path.join(nass_dir, 'nass_land_value_per_acre_county.csv'),
                    dtype={'STATE_FIPS_CODE': str, 'COUNTY_ANSI': str})
nass = nass[nass['VALUE'] != '(D)'].copy()
nass['VALUE'] = nass['VALUE'].str.replace(',', '').astype(float)
nass['YEAR'] = nass['YEAR'].astype(int)
nass = nass.dropna(subset=['STATE_FIPS_CODE', 'COUNTY_ANSI'])
nass['fips5'] = nass['STATE_FIPS_CODE'].str.zfill(2) + nass['COUNTY_ANSI'].str.zfill(3)

panel = geo.merge(nass[['fips5', 'YEAR', 'VALUE']], on='fips5', how='inner')
panel = panel.rename(columns={'YEAR': 'year', 'VALUE': 'landvalue'})
panel['log_landvalue'] = np.log(panel['landvalue'])
panel['log_population'] = np.log(panel['population'])
panel['log_land_area'] = np.log(panel['land_area_sqmi'])
panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= THRESH).astype(int)

for st in ['MT', 'CO']:
    sub = panel[panel['state_abbr'] == st].copy()
    n_border = sub.loc[sub['border_70mi'] == 1, 'fips5'].nunique()
    n_interior = sub.loc[sub['border_70mi'] == 0, 'fips5'].nunique()
    print(f"\n{'='*70}\n{st}: {sub['fips5'].nunique()} counties total "
          f"({n_border} border, {n_interior} interior), years {sorted(sub['year'].unique())}\n{'='*70}")
    if n_border == 0:
        print(f"  No border counties for {st} at {THRESH}mi threshold -- cannot test.")
        continue
    for name, formula in {
        'no controls': 'log_landvalue ~ border_70mi',
        '+log(pop)+year FE': 'log_landvalue ~ border_70mi + log_population + C(year)',
        '+log(pop)+land area+year FE': 'log_landvalue ~ border_70mi + log_population + log_land_area + C(year)',
    }.items():
        try:
            m = smf.ols(formula, data=sub).fit(cov_type='HC1')
            b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
            print(f"  {name:32s} b={b:+.4f} se={se:.4f} p={p:.4f} n={int(m.nobs)}")
        except Exception as e:
            print(f"  {name:32s} failed: {e}")
