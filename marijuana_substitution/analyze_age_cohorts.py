# ============================================================
# AGE-COHORT ANALYSIS: does the Design-A marijuana/alcohol
# switcher-vs-control comparison differ by age of offender?
# Same treatment design as analyze_partA.py (switcher/continuous
# distance, never-legal-state counties, 2020->2023 first
# difference, SE clustered by state), applied separately within
# each age bin using NIBRS Arrestee-Level microdata instead of
# the SRS-based agency pull.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

AGE_BINS = ['<21', '21-30', '31-45', '46+']

def z5(s):
    return s.astype(str).str.zfill(5)

panel = pd.read_stata(os.path.join(PARENT, 'marijuana_new', 'data', 'analysis_panel.dta'))
panel['fips5'] = z5(panel['fips5'])

meta = (panel[panel['year'] == 2020]
        [['fips5', 'switcher', 'never_legal', 'sample_grp', 'state_abbr', 'population',
          'dist_change_mi', 'log_dist_2020', 'log_dist_2023']]
        .drop_duplicates('fips5'))
meta['d_log_dist'] = meta['log_dist_2023'] - meta['log_dist_2020']
meta['d_dist_mi'] = meta['dist_change_mi']
meta['d_dist_100mi'] = meta['d_dist_mi'] / 100.0

u2020 = set(z5(pd.read_csv(os.path.join(DATA, 'nibrs_universe_2020.csv'))['fips5']))
u2023 = set(z5(pd.read_csv(os.path.join(DATA, 'nibrs_universe_2023.csv'))['fips5']))
universe = u2020 & u2023
print(f"NIBRS county universe: 2020={len(u2020)} 2023={len(u2023)} both={len(universe)}")

meta_u = meta[meta['fips5'].isin(universe) & meta['population'].notna() & (meta['population'] > 0)].copy()
print(f"Meta counties with population + NIBRS coverage both years: {len(meta_u)}")

age20 = pd.read_csv(os.path.join(DATA, 'age_arrests_2020.csv'))
age20['fips5'] = z5(age20['fips5'])
age23 = pd.read_csv(os.path.join(DATA, 'age_arrests_2023.csv'))
age23['fips5'] = z5(age23['fips5'])

def ols_cl(y, X, groups, xname):
    g = pd.Categorical(groups).codes
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
    return float(res.params[xname]), float(res.bse[xname]), float(res.pvalues[xname]), int(res.nobs)

all_results = []
all_panels = []

for age_bin in AGE_BINS:
    a20 = (age20[age20['age_bin'] == age_bin][['fips5', 'mj', 'alc']]
           .rename(columns={'mj': 'mj_2020', 'alc': 'alc_2020'}))
    a23 = (age23[age23['age_bin'] == age_bin][['fips5', 'mj', 'alc']]
           .rename(columns={'mj': 'mj_2023', 'alc': 'alc_2023'}))

    grid = meta_u[['fips5']].drop_duplicates()
    grid = grid.merge(a20, on='fips5', how='left').merge(a23, on='fips5', how='left')
    for c in ['mj_2020', 'mj_2023', 'alc_2020', 'alc_2023']:
        grid[c] = grid[c].fillna(0)

    full = meta_u.merge(grid, on='fips5', how='inner')

    for base in ['mj', 'alc']:
        full[f'{base}_rate20'] = full[f'{base}_2020'] / (full['population'] / 1e5)
        full[f'{base}_rate23'] = full[f'{base}_2023'] / (full['population'] / 1e5)
        full[f'd_{base}'] = full[f'{base}_rate23'] - full[f'{base}_rate20']

    main = full[full['sample_grp'] != 'Always near rec border'].copy()
    n_sw = int(main[main.switcher == 1]['fips5'].nunique())
    n_ct = int(main[main.switcher == 0]['fips5'].nunique())

    main['age_bin'] = age_bin
    all_panels.append(main)

    for base, label in [('mj', 'Marijuana'), ('alc', 'Alcohol')]:
        b, se, p, n = ols_cl(main[f'd_{base}'], main[['switcher']], main['state_abbr'], 'switcher')
        all_results.append(dict(age_bin=age_bin, outcome=label, spec='switcher', coef=b, se=se, p=p, n=n,
                                 n_switchers=n_sw, n_controls=n_ct))
        bm, sem, pm, nm = ols_cl(main[f'd_{base}'], main[['d_dist_100mi']], main['state_abbr'], 'd_dist_100mi')
        all_results.append(dict(age_bin=age_bin, outcome=label, spec='dist_100mi', coef=bm, se=sem, p=pm, n=nm,
                                 n_switchers=n_sw, n_controls=n_ct))
        b2, se2, p2, n2 = ols_cl(main[f'd_{base}'], main[['d_log_dist']], main['state_abbr'], 'd_log_dist')
        all_results.append(dict(age_bin=age_bin, outcome=label, spec='log_dist', coef=b2, se=se2, p=p2, n=n2,
                                 n_switchers=n_sw, n_controls=n_ct))

    print(f"\n=== Age bin {age_bin}: N={len(main)} (switchers={n_sw}, controls={n_ct}) ===")
    for base, label in [('mj', 'Marijuana'), ('alc', 'Alcohol')]:
        sub = [r for r in all_results if r['age_bin'] == age_bin and r['outcome'] == label]
        for r in sub:
            print(f"  {label:9s} {r['spec']:12s} b={r['coef']:+.3f} se={r['se']:.3f} p={r['p']:.3f}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(TABLES, 'tC1_age_regressions.csv'), index=False)
print(f"\nSaved {os.path.join(TABLES, 'tC1_age_regressions.csv')}")

combined_panel = pd.concat(all_panels, ignore_index=True)
combined_panel.to_csv(os.path.join(DATA, 'partC_age_panel.csv'), index=False)
print(f"Saved {os.path.join(DATA, 'partC_age_panel.csv')} ({len(combined_panel)} rows)")
