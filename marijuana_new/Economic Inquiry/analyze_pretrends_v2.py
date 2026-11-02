# ============================================================
# UNIFIED pre-trends / event-study check, built directly on
# stacked_panel_rec_full.csv -- the EXACT panel behind Table 1 and
# Table 2 -- rather than a separately-reconstructed panel. This
# replaces analyze_pretrends.py, which built its own panel from
# distance_by_year_rec.csv and produced numbers that didn't quite
# match the main-results panel (e.g. a "no controls" switch
# coefficient of -3.33 vs. Table 2's -3.135), a discrepancy that
# should not exist since both are supposed to describe the same
# raw comparison.
#
# Event time is recovered from the panel itself: for each county,
# find the transition (if any) where switcher==1 -- that is its
# switch transition (event_time=0). All of that county's OTHER
# rows (which are, by construction, pre-switch transitions, since
# the panel already drops counties after they switch) get
# event_time = (their transition's position) - (switch transition's
# position). Counties with no switcher==1 row anywhere are true
# never-switchers and serve as the omitted reference group.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

TRANSITIONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
T_INDEX = {t: i for i, t in enumerate(TRANSITIONS)}

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['t_idx'] = s['transition'].map(T_INDEX)

switch_idx = s.loc[s['switcher'] == 1, ['fips5', 't_idx']].rename(columns={'t_idx': 'switch_idx'})
n_dup = switch_idx['fips5'].duplicated().sum()
print(f"Counties with more than one switch row (should be 0): {n_dup}")

s = s.merge(switch_idx, on='fips5', how='left')
s['event_time'] = s['t_idx'] - s['switch_idx']  # NaN for never-switchers

def bucket(row):
    if pd.isna(row['switch_idx']):
        return 'control'
    if row['event_time'] <= -2:
        return 'lead2plus'
    if row['event_time'] == -1:
        return 'lead1'
    if row['event_time'] == 0:
        return 'switch0'
    return 'drop'  # should not occur: panel drops counties after they switch

s['bucket'] = s.apply(bucket, axis=1)
print("\nRow counts by bucket (unified panel):")
print(s['bucket'].value_counts())
assert (s['bucket'] == 'drop').sum() == 0, "Found post-switch rows -- panel construction assumption violated"

for b in ['lead1', 'lead2plus', 'switch0']:
    s[b] = (s['bucket'] == b).astype(float)
s['g'] = pd.Categorical(s['state_abbr']).codes

# baseline (2020) overdose-rate level, for the mean-reversion control
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)

print("\n=== Raw (transition FE only -- already present, no additional control) ===")
m_raw = smf.ols('d_od ~ lead2plus + lead1 + switch0 + C(transition)', data=s).fit(
    cov_type='cluster', cov_kwds={'groups': s['g']})
for t in ['lead2plus', 'lead1', 'switch0']:
    print(f"  {t}: b={m_raw.params[t]:+.4f} p={m_raw.pvalues[t]:.4f}  n={int((s['bucket']==t).sum())}")
f_raw = m_raw.f_test('lead2plus = 0, lead1 = 0')
print(f"  JOINT F-test (leads=0): F={float(f_raw.fvalue):.4f} p={float(f_raw.pvalue):.4f}")

print("\n=== + baseline-rate control (preferred) ===")
sub = s.dropna(subset=['baseline_rate'])
m_adj = smf.ols('d_od ~ lead2plus + lead1 + switch0 + baseline_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['g']})
for t in ['lead2plus', 'lead1', 'switch0']:
    print(f"  {t}: b={m_adj.params[t]:+.4f} p={m_adj.pvalues[t]:.4f}  n={int((sub['bucket']==t).sum())}")
f_adj = m_adj.f_test('lead2plus = 0, lead1 = 0')
print(f"  JOINT F-test (leads=0): F={float(f_adj.fvalue):.4f} p={float(f_adj.pvalue):.4f}")
print(f"  N (baseline-rate available): {int(m_adj.nobs)}")

# save for the figure script and for a division x transition cross-check
s.to_csv(os.path.join(DATA, 'pretrends_panel_rec_v2.csv'), index=False)

out = dict(
    raw={t: dict(coef=float(m_raw.params[t]), se=float(m_raw.bse[t]), p=float(m_raw.pvalues[t])) for t in ['lead2plus','lead1','switch0']},
    raw_joint_f=dict(F=float(f_raw.fvalue), p=float(f_raw.pvalue)),
    adjusted={t: dict(coef=float(m_adj.params[t]), se=float(m_adj.bse[t]), p=float(m_adj.pvalues[t])) for t in ['lead2plus','lead1','switch0']},
    adjusted_joint_f=dict(F=float(f_adj.fvalue), p=float(f_adj.pvalue)),
    n_adjusted=int(m_adj.nobs),
)
with open(os.path.join(DATA, 'pretrends_v2.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved pretrends_panel_rec_v2.csv, pretrends_v2.json")
