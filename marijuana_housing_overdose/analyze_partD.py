# ============================================================
# PART D ANALYSIS: Housing (Zillow ZHVI/ZORI) and drug-overdose
# mortality (CDC VSRR provisional county 12-month-ending counts)
# spillovers from proximity to a newly-legal recreational-
# marijuana border. Same switcher/control county design and
# continuous distance-to-border treatment as the companion
# marijuana_substitution project's Design A.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

def z5(s):
    return s.astype(str).str.zfill(5)

# ── Treatment panel (reused from the companion project) ─────────
# Switcher/control cutoff redefined at 20 miles (down from the companion
# project's 100-mile default), falling back to 50 miles per the requested
# rule since 20mi leaves only 32 switcher counties nationally -- too few
# for reliable cluster-robust inference once split across three outcomes
# with very different coverage (32 -> single digits once intersected with
# ZORI's ~15% county coverage, for instance). 50mi gives 212 switchers.
THRESH = 50

panel = pd.read_stata(os.path.join(PARENT, 'marijuana_new', 'data', 'analysis_panel.dta'))
panel['fips5'] = z5(panel['fips5'])
meta = (panel[panel['year'] == 2020]
        [['fips5', 'never_legal', 'state_abbr', 'population',
          'dist_rec_2020_mi', 'dist_rec_2023_mi', 'dist_change_mi',
          'log_dist_2020', 'log_dist_2023', 'lat', 'lon']]
        .drop_duplicates('fips5'))

always_near = (meta['dist_rec_2020_mi'] < THRESH) & (meta['dist_rec_2023_mi'] < THRESH)
switcher = (meta['dist_rec_2020_mi'] >= THRESH) & (meta['dist_rec_2023_mi'] < THRESH)
meta['switcher'] = switcher.astype(int)
meta['sample_grp'] = np.select(
    [always_near, switcher], ['Always near rec border', 'Switcher'], default='Control')

meta['d_log_dist'] = meta['log_dist_2023'] - meta['log_dist_2020']
meta['d_dist_thresh'] = meta['dist_change_mi'] / THRESH
print(f"Meta panel: {len(meta)} counties. Switcher cutoff = {THRESH} mi -- "
      f"always_near={always_near.sum()}, switcher={switcher.sum()}, control={(~always_near & ~switcher).sum()}")

# ── Housing: Zillow ZHVI / ZORI ──────────────────────────────────
zhvi = pd.read_csv(os.path.join(DATA, 'zhvi_county.csv'), dtype={'StateCodeFIPS': str, 'MunicipalCodeFIPS': str})
zhvi = zhvi[zhvi['RegionType'] == 'county'].copy()
zhvi['fips5'] = zhvi['StateCodeFIPS'].str.zfill(2) + zhvi['MunicipalCodeFIPS'].str.zfill(3)
zhvi = zhvi[['fips5', '2020-12-31', '2023-12-31']].rename(
    columns={'2020-12-31': 'zhvi_2020', '2023-12-31': 'zhvi_2023'})

zori = pd.read_csv(os.path.join(DATA, 'zori_county.csv'), dtype={'StateCodeFIPS': str, 'MunicipalCodeFIPS': str})
zori = zori[zori['RegionType'] == 'county'].copy()
zori['fips5'] = zori['StateCodeFIPS'].str.zfill(2) + zori['MunicipalCodeFIPS'].str.zfill(3)
zori = zori[['fips5', '2020-12-31', '2023-12-31']].rename(
    columns={'2020-12-31': 'zori_2020', '2023-12-31': 'zori_2023'})

print(f"ZHVI counties: {zhvi['fips5'].nunique()}  ZORI counties: {zori['fips5'].nunique()}")

# ── Overdose mortality: CDC VSRR provisional county 12-month-ending counts ──
with open(os.path.join(DATA, 'overdose_county_raw.json'), encoding='utf-8-sig') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()
n_suppressed_20 = od_raw[od_raw['year'] == '2020']['provisional_drug_overdose'].isna().sum()
n_suppressed_23 = od_raw[od_raw['year'] == '2023']['provisional_drug_overdose'].isna().sum()
print(f"Overdose: {len(od)} counties in raw file. "
      f"Suppressed (<10 deaths, NCHS confidentiality rule): 2020={n_suppressed_20}, 2023={n_suppressed_23}")

# ── Merge everything ─────────────────────────────────────────────
full = meta.merge(zhvi, on='fips5', how='left').merge(zori, on='fips5', how='left').merge(od, on='fips5', how='left')
full = full[full['population'].notna() & (full['population'] > 0)].copy()

full['d_zhvi_log'] = np.log(full['zhvi_2023']) - np.log(full['zhvi_2020'])
full['d_zori_log'] = np.log(full['zori_2023']) - np.log(full['zori_2020'])
full['od_rate20'] = full['od_2020'] / (full['population'] / 1e5)
full['od_rate23'] = full['od_2023'] / (full['population'] / 1e5)
full['d_od'] = full['od_rate23'] - full['od_rate20']

main = full[full['sample_grp'] != 'Always near rec border'].copy()
n_sw = int(main[main.switcher == 1]['fips5'].nunique())
n_ct = int(main[main.switcher == 0]['fips5'].nunique())
print(f"\nMain sample: {len(main)} counties (switchers={n_sw}, controls={n_ct})")
for col, label in [('d_zhvi_log', 'ZHVI (log)'), ('d_zori_log', 'ZORI (log)'), ('d_od', 'Overdose rate')]:
    print(f"  Non-missing {label}: {main[col].notna().sum()} / {len(main)}")

def ols_cl(y, X, groups, xname):
    ok = y.notna() & X[xname].notna() & groups.notna()
    y, X, groups = y[ok], X[ok], groups[ok]
    g = pd.Categorical(groups).codes
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
    return float(res.params[xname]), float(res.bse[xname]), float(res.pvalues[xname]), int(res.nobs)

results = {}
rows = []
for outcome_col, label in [('d_zhvi_log', 'Home value (log)'), ('d_zori_log', 'Rent (log)'), ('d_od', 'Overdose rate per 100k')]:
    avail = main[main[outcome_col].notna()]
    n_sw_avail = int(avail[avail.switcher == 1]['fips5'].nunique())
    n_ct_avail = int(avail[avail.switcher == 0]['fips5'].nunique())
    for spec_name, xcol in [('switcher', 'switcher'), ('dist_thresh', 'd_dist_thresh'), ('log_dist', 'd_log_dist')]:
        b, se, p, n = ols_cl(main[outcome_col], main[[xcol]], main['state_abbr'], xcol)
        key = f"{outcome_col}__{spec_name}"
        results[key] = (b, se, p, n)
        rows.append(dict(outcome=label, spec=spec_name, coef=b, se=se, p=p, n=n,
                          n_switchers=n_sw_avail, n_controls=n_ct_avail))
        print(f"{label:24s} {spec_name:12s} b={b:+.5f} se={se:.5f} p={p:.3f} n={n} "
              f"(sw={n_sw_avail}, ct={n_ct_avail})")

pd.Series(results).to_json(os.path.join(DATA, 'partD_results.json'))
pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'tD1_regressions.csv'), index=False)
main.to_csv(os.path.join(DATA, 'partD_county_panel.csv'), index=False)
print(f"\nSaved partD_county_panel.csv ({len(main)} rows), partD_results.json, tD1_regressions.csv")
