# ============================================================
# Full event-study check (+/-3 years around the switch), built
# directly on distance_by_year_rec_full.csv -- the same source
# persistence.py and the diff-in-disc check use, which (unlike
# stacked_panel_rec_full.csv) is NOT deduplicated of post-switch
# rows, so it supports leads AND lags in one single, consistently-
# built panel. This avoids the leads/lags-built-on-different-
# panels mismatch that was caught and fixed in pretrends_v2.py.
#
# Event time = transition_end_year - switch_year (NaN for counties
# that never cross the 50-mile threshold in 2020-2025; these are
# the omitted reference/control group, as in every other event-
# study check in this project). Bins are ENDPOINT-BINNED at the
# requested +/-3 window (event_time <= -3 pooled into "-3", >= +3
# pooled into "+3") since raw per-year tails are thin at the edges
# of a 5-transition panel.
#
# Standard errors are clustered by COUNTY (fips5), not state --
# a stricter/finer clustering than every other regression in this
# project (which cluster by state), per this round's explicit
# request.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

THRESH = 50
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

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

# always-near (already <50mi at the Dec-2020 baseline) excluded, matches main design's county universe
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
    sub['od_2020'] = sub['od_2020'] if 'od_2020' in sub.columns else np.nan
    rows.append(sub[['fips5', 'state_abbr', 'population', 'transition', 'event_time', 'd_od', 'od_2020']])

stacked = pd.concat(rows, ignore_index=True)
stacked = stacked[stacked['d_od'].notna() & (stacked['population'] > 0)].copy()

def bin_label(et):
    # patsy chokes on '-' in identifiers (parses as subtraction), so
    # use et_m{k} / et_p{k} instead of a literal signed integer
    if et < 0:
        return f'et_m{abs(et)}'
    return f'et_p{et}' if et > 0 else 'et_0'

def bucket(et):
    if pd.isna(et):
        return 'control'
    et = int(et)
    et = max(-3, min(3, et))  # endpoint-bin at the requested +/-3 window
    return bin_label(et)

stacked['bucket'] = stacked['event_time'].apply(bucket)
print("\nRow counts by bucket:")
print(stacked['bucket'].value_counts().sort_index())

EVENT_BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
for b in EVENT_BINS:
    stacked[b] = (stacked['bucket'] == b).astype(float)

stacked['baseline_rate'] = stacked['od_2020'] / (stacked['population'] / 1e5)
stacked['county_g'] = pd.Categorical(stacked['fips5']).codes  # cluster group: COUNTY, not state

rhs_terms = ' + '.join(EVENT_BINS)

print("\n=== Raw (transition FE only), clustered by county ===")
m_raw = smf.ols(f'd_od ~ {rhs_terms} + C(transition)', data=stacked).fit(
    cov_type='cluster', cov_kwds={'groups': stacked['county_g']})
for b in EVENT_BINS:
    print(f"  {b}: b={m_raw.params[b]:+.4f} se={m_raw.bse[b]:.4f} p={m_raw.pvalues[b]:.4f}  n={int((stacked['bucket']==b).sum())}")

print("\n=== + baseline-rate control (preferred, matches Table 1), clustered by county ===")
sub = stacked.dropna(subset=['baseline_rate'])
m_adj = smf.ols(f'd_od ~ {rhs_terms} + baseline_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})
for b in EVENT_BINS:
    print(f"  {b}: b={m_adj.params[b]:+.4f} se={m_adj.bse[b]:.4f} p={m_adj.pvalues[b]:.4f}  n={int((sub['bucket']==b).sum())}")
f_leads = m_adj.f_test(', '.join([f'{b} = 0' for b in ['et_m3', 'et_m2', 'et_m1']]))
print(f"  Joint F-test (all leads = 0): F={float(f_leads.fvalue):.4f} p={float(f_leads.pvalue):.4f}")

stacked.to_csv(os.path.join(DATA, 'eventstudy_full_panel_rec.csv'), index=False)
out = dict(
    raw={b: dict(coef=float(m_raw.params[b]), se=float(m_raw.bse[b]), p=float(m_raw.pvalues[b]),
                 n=int((stacked['bucket']==b).sum())) for b in EVENT_BINS},
    adjusted={b: dict(coef=float(m_adj.params[b]), se=float(m_adj.bse[b]), p=float(m_adj.pvalues[b]),
                       n=int((sub['bucket']==b).sum())) for b in EVENT_BINS},
    joint_f_leads_zero=dict(F=float(f_leads.fvalue), p=float(f_leads.pvalue)),
    n_adjusted=int(m_adj.nobs),
    cluster='county (fips5)',
)
with open(os.path.join(DATA, 'eventstudy_full_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved eventstudy_full_panel_rec.csv, eventstudy_full_results.json")
