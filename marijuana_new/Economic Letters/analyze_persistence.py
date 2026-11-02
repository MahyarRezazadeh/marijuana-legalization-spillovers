# ============================================================
# Persistence check: does the switcher effect continue, flatten,
# or reverse in the transitions AFTER a county crosses the
# 50-mile threshold? The main stacked design (build_multiperiod_
# panel.py) deliberately DROPS a county from all later transitions
# once it switches, to avoid the already-treated-as-control bias
# -- which means it has no built-in way to see what happens after
# switching. This script rebuilds the panel WITHOUT that drop,
# purely to construct lag (post-switch) event-time indicators,
# using the same never-switching counties as the reference group
# and the same event-time logic as the pre-trends (leads) check,
# just extended to positive event-time.
#
# Each lag bucket is a SEQUENTIAL one-year transition (lag1 = the
# first one-year change after switching, lag2 = the second, etc.),
# not a cumulative change since switching -- this asks whether the
# year-over-year decline continues after the initial switch
# (lags also negative/significant), flattens (a one-time level
# shift that then persists, lags ~0), or reverses (lags positive,
# effect fading).
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

THRESH = 50
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TRANSITIONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']

dist = pd.read_csv(os.path.join(DATA, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()
dist = dist.merge(od, on='fips5', how='left')

# always-near (already <50mi at the Dec-2020 baseline) are excluded, as in the main design
dist = dist[dist['dist_2020'] >= THRESH].copy()
print(f"County universe (excludes {2331 - len(dist)} always-near): {len(dist)}")

def find_switch_year(row):
    for y in YEARS[1:]:
        if row[f'dist_{y-1}'] >= THRESH and row[f'dist_{y}'] < THRESH:
            return y
    return np.nan

dist['switch_year'] = dist.apply(find_switch_year, axis=1)
n_switchers = dist['switch_year'].notna().sum()
print(f"Eventual switchers: {n_switchers}, never-switchers: {len(dist) - n_switchers}")

rows = []
for t in YEARS[1:]:
    t0 = t - 1
    sub = dist.copy()
    sub['transition'] = f'{t0}-{t}'
    sub['event_time'] = t - sub['switch_year']  # NaN for never-switchers
    sub['od_rate_pre'] = sub[f'od_{t0}'] / (sub['population'] / 1e5)
    sub['od_rate_post'] = sub[f'od_{t}'] / (sub['population'] / 1e5)
    sub['d_od'] = sub['od_rate_post'] - sub['od_rate_pre']
    rows.append(sub[['fips5', 'state_abbr', 'population', 'transition', 'event_time', 'd_od']])

stacked = pd.concat(rows, ignore_index=True)
stacked = stacked[stacked['d_od'].notna() & (stacked['population'] > 0)].copy()

def bucket(et):
    if pd.isna(et):
        return 'control'
    if et == 0:
        return 'switch0'
    if et == 1:
        return 'lag1'
    if et >= 2:
        return 'lag2plus'
    return 'lead'  # negative event-time, not used in this check

stacked['bucket'] = stacked['event_time'].apply(bucket)
print("\nRow counts by bucket:")
print(stacked['bucket'].value_counts())

stacked = stacked[stacked['bucket'] != 'lead'].copy()
for b in ['switch0', 'lag1', 'lag2plus']:
    stacked[b] = (stacked['bucket'] == b).astype(float)
stacked['g'] = pd.Categorical(stacked['state_abbr']).codes

# baseline-rate (mean-reversion) control, same as the preferred main specification
od2020 = od_raw[od_raw['year'] == 2020][['fips5', 'provisional_drug_overdose']].rename(
    columns={'provisional_drug_overdose': 'od_2020_raw'})
stacked = stacked.merge(dist[['fips5', 'population']].drop_duplicates(), on='fips5', how='left', suffixes=('', '_dup'))
stacked = stacked.merge(od2020, on='fips5', how='left')
stacked['baseline_rate'] = stacked['od_2020_raw'] / (stacked['population'] / 1e5)

print("\n=== Raw (transition FE only) ===")
m_raw = smf.ols('d_od ~ switch0 + lag1 + lag2plus + C(transition)', data=stacked).fit(
    cov_type='cluster', cov_kwds={'groups': stacked['g']})
for t in ['switch0', 'lag1', 'lag2plus']:
    print(f"  {t}: b={m_raw.params[t]:+.4f} se={m_raw.bse[t]:.4f} p={m_raw.pvalues[t]:.4f}  n={int((stacked['bucket']==t).sum())}")

print("\n=== + baseline-rate control (preferred) ===")
sub = stacked.dropna(subset=['baseline_rate'])
m_adj = smf.ols('d_od ~ switch0 + lag1 + lag2plus + baseline_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['g']})
for t in ['switch0', 'lag1', 'lag2plus']:
    print(f"  {t}: b={m_adj.params[t]:+.4f} se={m_adj.bse[t]:.4f} p={m_adj.pvalues[t]:.4f}  n={int((sub['bucket']==t).sum())}")
f_adj = m_adj.f_test('lag1 = 0, lag2plus = 0')
print(f"  Joint F-test (lags both = 0): F={float(f_adj.fvalue):.4f} p={float(f_adj.pvalue):.4f}")
f_adj2 = m_adj.f_test('switch0 = lag1, switch0 = lag2plus')
print(f"  Joint F-test (lags = switch0, i.e. fully persists): F={float(f_adj2.fvalue):.4f} p={float(f_adj2.pvalue):.4f}")

stacked.to_csv(os.path.join(DATA, 'persistence_panel_rec.csv'), index=False)
out = dict(
    raw={t: dict(coef=float(m_raw.params[t]), se=float(m_raw.bse[t]), p=float(m_raw.pvalues[t])) for t in ['switch0','lag1','lag2plus']},
    adjusted={t: dict(coef=float(m_adj.params[t]), se=float(m_adj.bse[t]), p=float(m_adj.pvalues[t])) for t in ['switch0','lag1','lag2plus']},
    joint_f_lags_zero=dict(F=float(f_adj.fvalue), p=float(f_adj.pvalue)),
    joint_f_lags_equal_switch=dict(F=float(f_adj2.fvalue), p=float(f_adj2.pvalue)),
    n_adjusted=int(m_adj.nobs),
)
with open(os.path.join(DATA, 'persistence_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved persistence_panel_rec.csv, persistence_results.json")
