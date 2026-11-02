# ============================================================
# Reruns the switcher + continuous overdose-mortality regressions
# on the rebuilt treatment panel (new legalization-timing data).
# Same outcome (CDC VSRR county overdose rate, 2020 vs 2023) and
# same design (binary <50mi switcher; continuous per-mile
# distance change) as before.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
MARIJUANA_NEW = os.path.dirname(BASE)
ROOT = os.path.dirname(MARIJUANA_NEW)
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
HOUSING_DATA = os.path.join(ROOT, 'marijuana_housing_overdose', 'data')

meta = pd.read_csv(os.path.join(DATA, 'treatment_panel_new.csv'), dtype={'fips5': str})
meta['d_dist_mi'] = meta['dist_change_mi']  # already miles, raw per-mile coefficient
meta['d_log_dist'] = np.log(meta['dist_2023_mi'].clip(lower=0.1)) - np.log(meta['dist_2020_mi'].clip(lower=0.1))

with open(os.path.join(HOUSING_DATA, 'overdose_county_raw.json'), encoding='utf-8-sig') as f:
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
print(f"Main sample: {len(main)} counties. Non-missing overdose data: {len(avail)} "
      f"(switchers={n_sw}, controls={n_ct})")

def ols_cl(y, X, groups, xname):
    ok = y.notna() & X[xname].notna() & groups.notna()
    y, X, groups = y[ok], X[ok], groups[ok]
    g = pd.Categorical(groups).codes
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
    return float(res.params[xname]), float(res.bse[xname]), float(res.pvalues[xname]), int(res.nobs)

results = {}
rows = []
for spec_name, xcol in [('switcher', 'switcher'), ('dist_mile', 'd_dist_mi'), ('log_dist', 'd_log_dist')]:
    b, se, p, n = ols_cl(main['d_od'], main[[xcol]], main['state_abbr'], xcol)
    results[spec_name] = (b, se, p, n)
    rows.append(dict(spec=spec_name, coef=b, se=se, p=p, n=n, n_switchers=n_sw, n_controls=n_ct))
    print(f"{spec_name:12s} b={b:+.5f} se={se:.5f} p={p:.4f} n={n}")

os.makedirs(TABLES, exist_ok=True)
pd.Series(results).to_json(os.path.join(DATA, 'overdose_results_new.json'))
pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'tE1_overdose_regressions_new.csv'), index=False)
main.to_csv(os.path.join(DATA, 'overdose_county_panel_new.csv'), index=False)

n_suppressed_20 = od_raw[od_raw['year'] == '2020']['provisional_drug_overdose'].isna().sum()
n_suppressed_23 = od_raw[od_raw['year'] == '2023']['provisional_drug_overdose'].isna().sum()
summary = pd.DataFrame([
    {'Variable': 'Overdose rate per 100k (2020, mean)', 'Value': f"{main['od_rate20'].mean():.2f}"},
    {'Variable': 'Overdose rate per 100k (2023, mean)', 'Value': f"{main['od_rate23'].mean():.2f}"},
    {'Variable': 'Switcher counties, full panel (N)', 'Value': int((meta['switcher']==1).sum())},
    {'Variable': 'Control counties, full panel (N)', 'Value': int((meta['sample_grp']=='Control').sum())},
    {'Variable': 'Switcher counties, non-suppressed (N)', 'Value': n_sw},
    {'Variable': 'Control counties, non-suppressed (N)', 'Value': n_ct},
])
summary.to_csv(os.path.join(TABLES, 'tE0_overdose_summary_new.csv'), index=False)
print(f"\nSaved overdose_county_panel_new.csv ({len(main)} rows), overdose_results_new.json, "
      f"tE1_overdose_regressions_new.csv, tE0_overdose_summary_new.csv")
