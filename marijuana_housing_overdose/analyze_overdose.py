# ============================================================
# OVERDOSE-ONLY ANALYSIS for the Economic Inquiry-formatted
# paper (main1). Two co-primary treatment specs: continuous
# distance (per 50mi) and a crude/binary switcher indicator
# (distance < 50mi in 2023, >=50mi in 2020). Log-distance kept
# as a robustness check. Reuses the same 50mi-threshold county
# panel built in analyze_partD.py.
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
meta['sample_grp'] = np.select([always_near, switcher], ['Always near rec border', 'Switcher'], default='Control')
meta['d_log_dist'] = meta['log_dist_2023'] - meta['log_dist_2020']
meta['d_dist_thresh'] = meta['dist_change_mi'] / THRESH
print(f"Meta panel: {len(meta)} counties. always_near={always_near.sum()}, "
      f"switcher={switcher.sum()}, control={(~always_near & ~switcher).sum()}")

with open(os.path.join(DATA, 'overdose_county_raw.json'), encoding='utf-8-sig') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()

full = meta.merge(od, on='fips5', how='left')
full = full[full['population'].notna() & (full['population'] > 0)].copy()
full['od_rate20'] = full['od_2020'] / (full['population'] / 1e5)
full['od_rate23'] = full['od_2023'] / (full['population'] / 1e5)
full['d_od'] = full['od_rate23'] - full['od_rate20']

main = full[full['sample_grp'] != 'Always near rec border'].copy()
avail = main[main['d_od'].notna()]
n_sw = int(avail[avail.switcher == 1]['fips5'].nunique())
n_ct = int(avail[avail.switcher == 0]['fips5'].nunique())
print(f"Main sample (non-missing overdose data): {len(avail)} counties (switchers={n_sw}, controls={n_ct})")

def ols_cl(y, X, groups, xname):
    ok = y.notna() & X[xname].notna() & groups.notna()
    y, X, groups = y[ok], X[ok], groups[ok]
    g = pd.Categorical(groups).codes
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
    return float(res.params[xname]), float(res.bse[xname]), float(res.pvalues[xname]), int(res.nobs)

results = {}
rows = []
for spec_name, xcol in [('switcher', 'switcher'), ('dist_thresh', 'd_dist_thresh'), ('log_dist', 'd_log_dist')]:
    b, se, p, n = ols_cl(main['d_od'], main[[xcol]], main['state_abbr'], xcol)
    results[spec_name] = (b, se, p, n)
    rows.append(dict(spec=spec_name, coef=b, se=se, p=p, n=n, n_switchers=n_sw, n_controls=n_ct))
    print(f"{spec_name:12s} b={b:+.5f} se={se:.5f} p={p:.4f} n={n}")

pd.Series(results).to_json(os.path.join(DATA, 'overdose_results.json'))
pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'tE1_overdose_regressions.csv'), index=False)
main.to_csv(os.path.join(DATA, 'overdose_county_panel.csv'), index=False)

n_suppressed_20 = od_raw[od_raw['year'] == '2020']['provisional_drug_overdose'].isna().sum()
n_suppressed_23 = od_raw[od_raw['year'] == '2023']['provisional_drug_overdose'].isna().sum()
summary = pd.DataFrame([
    {'Variable': 'Overdose rate per 100k (2020, mean)', 'Value': f"{main['od_rate20'].mean():.2f}"},
    {'Variable': 'Overdose rate per 100k (2023, mean)', 'Value': f"{main['od_rate23'].mean():.2f}"},
    {'Variable': 'Switcher counties, full panel (N)', 'Value': int(switcher.sum())},
    {'Variable': 'Control counties, full panel (N)', 'Value': int((~always_near & ~switcher).sum())},
    {'Variable': 'Switcher counties, non-suppressed (N)', 'Value': n_sw},
    {'Variable': 'Control counties, non-suppressed (N)', 'Value': n_ct},
    {'Variable': 'Counties suppressed nationally, 2020', 'Value': int(n_suppressed_20)},
    {'Variable': 'Counties suppressed nationally, 2023', 'Value': int(n_suppressed_23)},
])
summary.to_csv(os.path.join(TABLES, 'tE0_overdose_summary.csv'), index=False)
print(f"\nSaved overdose_county_panel.csv ({len(main)} rows), overdose_results.json, tE1_overdose_regressions.csv, tE0_overdose_summary.csv")
