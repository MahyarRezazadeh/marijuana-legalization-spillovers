# ============================================================
# SUPPLEMENTARY ANALYSIS: marijuana & alcohol arrest rates
# (SRS-style FBI CDE agency pull, from the companion
# marijuana_substitution project), rerun at the SAME 50-mile
# switcher/control threshold as the main overdose analysis
# (the companion project's own headline results use its
# original 100-mile threshold; this supplementary table exists
# purely for treatment-design consistency with the overdose
# result in this paper).
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
SUBST_DATA = os.path.join(PARENT, 'marijuana_substitution', 'data')

def z5(s):
    return s.astype(str).str.zfill(5)

THRESH = 50

panel = pd.read_stata(os.path.join(PARENT, 'marijuana_new', 'data', 'analysis_panel.dta'))
panel['fips5'] = z5(panel['fips5'])
meta = (panel[panel['year'] == 2020]
        [['fips5', 'never_legal', 'state_abbr', 'population',
          'dist_rec_2020_mi', 'dist_rec_2023_mi', 'dist_change_mi',
          'log_dist_2020', 'log_dist_2023']]
        .drop_duplicates('fips5'))

always_near = (meta['dist_rec_2020_mi'] < THRESH) & (meta['dist_rec_2023_mi'] < THRESH)
switcher = (meta['dist_rec_2020_mi'] >= THRESH) & (meta['dist_rec_2023_mi'] < THRESH)
meta['switcher'] = switcher.astype(int)
meta['sample_grp'] = np.select([always_near, switcher], ['Always near rec border', 'Switcher'], default='Control')
meta['d_log_dist'] = meta['log_dist_2023'] - meta['log_dist_2020']
meta['d_dist_thresh'] = meta['dist_change_mi'] / THRESH

arrests = pd.read_csv(os.path.join(SUBST_DATA, 'county_arrests_matched.csv'))
arrests['fips5'] = z5(arrests['fips5'])

full = meta.merge(arrests[['fips5', 'mj_2020', 'mj_2023', 'alc_2020', 'alc_2023']], on='fips5', how='left')
full = full[full['population'].notna() & (full['population'] > 0)].copy()
for base in ['mj', 'alc']:
    full[f'{base}_rate20'] = full[f'{base}_2020'] / (full['population'] / 1e5)
    full[f'{base}_rate23'] = full[f'{base}_2023'] / (full['population'] / 1e5)
    full[f'd_{base}'] = full[f'{base}_rate23'] - full[f'{base}_rate20']

main = full[full['sample_grp'] != 'Always near rec border'].copy()

def ols_cl(y, X, groups, xname):
    ok = y.notna() & X[xname].notna() & groups.notna()
    y, X, groups = y[ok], X[ok], groups[ok]
    g = pd.Categorical(groups).codes
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
    return float(res.params[xname]), float(res.bse[xname]), float(res.pvalues[xname]), int(res.nobs)

rows = []
for base, label in [('mj', 'Marijuana arrests'), ('alc', 'Alcohol arrests')]:
    avail = main[main[f'd_{base}'].notna()]
    n_sw = int(avail[avail.switcher == 1]['fips5'].nunique())
    n_ct = int(avail[avail.switcher == 0]['fips5'].nunique())
    for spec_name, xcol in [('switcher', 'switcher'), ('dist_thresh', 'd_dist_thresh'), ('log_dist', 'd_log_dist')]:
        b, se, p, n = ols_cl(main[f'd_{base}'], main[[xcol]], main['state_abbr'], xcol)
        rows.append(dict(outcome=label, spec=spec_name, coef=b, se=se, p=p, n=n, n_switchers=n_sw, n_controls=n_ct))
        print(f"{label:18s} {spec_name:12s} b={b:+.4f} se={se:.4f} p={p:.4f} n={n} (sw={n_sw}, ct={n_ct})")

pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'tE2_arrest_supplementary.csv'), index=False)
main.to_csv(os.path.join(DATA, 'arrest_supp_panel.csv'), index=False)
print(f"\nSaved tE2_arrest_supplementary.csv, arrest_supp_panel.csv ({len(main)} rows)")
