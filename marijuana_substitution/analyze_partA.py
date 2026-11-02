# ============================================================
# PART A ANALYSIS: Marijuana vs alcohol arrest trends in
# switcher vs control counties (never-legalizing states).
# Crosswalks agency-level pulls to county FIPS, merges with the
# companion project's analysis_panel.dta, runs first-difference
# regressions for both outcomes, looks for a substitution pattern.
# ============================================================
import os, re
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
DATA = os.path.join(BASE, 'data')

def norm_county(name):
    if pd.isna(name) or not str(name).strip():
        return None
    s = str(name).upper().strip()
    s = re.sub(r"[.']", '', s)
    s = re.sub(r'\s+', ' ', s)
    for suf in [' COUNTY', ' PARISH', ' BOROUGH', ' CENSUS AREA', ' CITY AND BOROUGH',
                ' MUNICIPALITY', ' CITY']:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()

# ── Load raw pulls ──────────────────────────────────────────────
agencies = pd.read_csv(os.path.join(DATA, 'agencies_raw.csv'))
arrests  = pd.read_csv(os.path.join(DATA, 'county_arrests_raw.csv'))
print(f"Agencies: {len(agencies)}  Arrest rows: {len(arrests)}")
print(arrests['status'].value_counts())

merged = agencies.merge(arrests, on='ori', how='inner')
ok = merged[merged['status'] == 'ok'].copy()
print(f"OK rows: {len(ok)} / {len(merged)} ({len(ok)/len(merged)*100:.1f}%)")

ok['n_counties'] = ok['counties_raw'].fillna('').apply(lambda s: len([c for c in str(s).split(',') if c.strip()]))
single = ok[ok['n_counties'] == 1].copy()
print(f"Single-county agencies: {len(single)} / {len(ok)}")
single['county_norm'] = single['counties_raw'].apply(norm_county)

# ── Census crosswalk from companion project's panel ─────────────
panel = pd.read_stata(os.path.join(PARENT, 'marijuana_new', 'data', 'analysis_panel.dta'))
panel['fips5'] = panel['fips5'].astype(str).str.zfill(5)
cw = panel[panel['year'] == 2023][['fips5', 'state_abbr', 'county_name_p2']].drop_duplicates()
cw = cw[cw['county_name_p2'].astype(str).str.strip() != '']
cw['county_norm'] = cw['county_name_p2'].apply(norm_county)
cw = cw.drop_duplicates(subset=['state_abbr', 'county_norm'])

joined = single.merge(cw[['fips5', 'state_abbr', 'county_norm']], on=['state_abbr', 'county_norm'], how='left')
matched = joined[joined['fips5'].notna()].copy()
print(f"Matched to FIPS: {len(matched)} / {len(joined)} ({len(matched)/len(joined)*100:.1f}%)")

for c in ['mj_2020', 'mj_2023', 'alc_2020', 'alc_2023']:
    matched[c] = pd.to_numeric(matched[c], errors='coerce')

county_arrests = (matched.groupby('fips5')
                  .agg(mj_2020=('mj_2020', 'sum'), mj_2023=('mj_2023', 'sum'),
                       alc_2020=('alc_2020', 'sum'), alc_2023=('alc_2023', 'sum'),
                       n_agencies=('ori', 'count'))
                  .reset_index())
county_arrests.to_csv(os.path.join(DATA, 'county_arrests_matched.csv'), index=False)

never_legal_counties = set(panel[panel['never_legal'] == 1]['fips5'].unique())
covered = never_legal_counties & set(county_arrests['fips5'])
print(f"Coverage: {len(covered)} / {len(never_legal_counties)} never-legal counties "
      f"({len(covered)/len(never_legal_counties)*100:.1f}%)")

# ── Merge with treatment panel (switcher, population, distance) ─
meta = (panel[panel['year'] == 2020]
        [['fips5', 'switcher', 'never_legal', 'sample_grp', 'state_abbr', 'population',
          'dist_rec_2020_mi', 'dist_rec_2023_mi', 'dist_change_mi', 'log_dist_2020', 'log_dist_2023',
          'lat', 'lon']]
        .drop_duplicates('fips5'))
meta['d_log_dist'] = meta['log_dist_2023'] - meta['log_dist_2020']
# Primary continuous treatment: raw change in miles to nearest rec border
# (dist_change_mi = dist_2023_mi - dist_2020_mi; negative = county became closer)
meta['d_dist_mi'] = meta['dist_change_mi']

full = meta.merge(county_arrests, on='fips5', how='inner')
full = full[full['population'].notna() & (full['population'] > 0)].copy()
print(f"\nFinal analysis panel: {len(full)} counties")

for base in ['mj', 'alc']:
    full[f'{base}_rate20'] = full[f'{base}_2020'] / (full['population'] / 1e5)
    full[f'{base}_rate23'] = full[f'{base}_2023'] / (full['population'] / 1e5)
    full[f'd_{base}'] = full[f'{base}_rate23'] - full[f'{base}_rate20']

main = full[full['sample_grp'] != 'Always near rec border'].copy()
n_sw = int(main[main.switcher == 1]['fips5'].nunique())
n_ct = int(main[main.switcher == 0]['fips5'].nunique())
print(f"Switchers: {n_sw}  Controls: {n_ct}")

def ols_cl(y, X, groups, xname):
    g = pd.Categorical(groups).codes
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
    return float(res.params[xname]), float(res.bse[xname]), float(res.pvalues[xname]), int(res.nobs)

results = {}
for base, label in [('mj', 'Marijuana'), ('alc', 'Alcohol')]:
    b, se, p, n = ols_cl(main[f'd_{base}'], main[['switcher']], main['state_abbr'], 'switcher')
    results[f'{base}_switcher'] = (b, se, p, n)
    # Primary continuous treatment: raw miles change (per-100-mile scaling for readability)
    main['d_dist_100mi'] = main['d_dist_mi'] / 100.0
    bm, sem, pm, nm = ols_cl(main[f'd_{base}'], main[['d_dist_100mi']], main['state_abbr'], 'd_dist_100mi')
    results[f'{base}_contdist_mi'] = (bm, sem, pm, nm)
    # Robustness: log-distance version (as used in the companion driving-spillover paper)
    b2, se2, p2, n2 = ols_cl(main[f'd_{base}'], main[['d_log_dist']], main['state_abbr'], 'd_log_dist')
    results[f'{base}_contdist_log'] = (b2, se2, p2, n2)
    print(f"{label}: switcher b={b:+.3f} se={se:.3f} p={p:.3f} | "
          f"d_dist(per 100mi) b={bm:+.3f} se={sem:.3f} p={pm:.3f} | "
          f"d_log_dist b={b2:+.3f} se={se2:.3f} p={p2:.3f}")

# Substitution metric: correlation between d_mj and d_alc across counties
corr = main[['d_mj', 'd_alc']].corr().iloc[0, 1]
print(f"\nCorrelation between d(marijuana arrest rate) and d(alcohol arrest rate) across counties: {corr:.3f}")

pd.Series(results).to_json(os.path.join(DATA, 'partA_results.json'))
main.to_csv(os.path.join(DATA, 'partA_county_panel.csv'), index=False)
print(f"\nSaved partA_county_panel.csv ({len(main)} rows) and partA_results.json")
