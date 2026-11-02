# ============================================================
# Full-time-span version of the event study: instead of the
# earlier +/-3-year window with endpoint-binned tails (et_m3 =
# "<=-3", et_p3 = ">=+3"), use the ENTIRE actual span the 2020-2025
# panel supports -- from the earliest possible transition (2021,
# relative to the latest possible switch year) to the latest
# possible transition (2025, relative to the earliest possible
# switch year) -- with a separate, un-pooled dummy for every event
# time that actually occurs in the data. No binning/truncation.
#
# Same panel construction, same never-switcher control group, same
# county-level clustering as analyze_eventstudy_full.py.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
PARENT_DATA = os.path.join(BASE, '..', 'data')

THRESH = 50
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

dist = pd.read_csv(os.path.join(PARENT_DATA, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
with open(os.path.join(PARENT_DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()
dist = dist.merge(od, on='fips5', how='left')

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
    sub['event_time'] = t - sub['switch_year']
    sub['od_rate_pre'] = sub[f'od_{t0}'] / (sub['population'] / 1e5)
    sub['od_rate_post'] = sub[f'od_{t}'] / (sub['population'] / 1e5)
    sub['d_od'] = sub['od_rate_post'] - sub['od_rate_pre']
    rows.append(sub[['fips5', 'state_abbr', 'population', 'transition', 'event_time', 'd_od', 'od_2020']])

stacked = pd.concat(rows, ignore_index=True)
stacked = stacked[stacked['d_od'].notna() & (stacked['population'] > 0)].copy()

# full actual span, no binning: start = earliest event_time in the (filtered)
# data, end = latest event_time in the data (naturally -4..+4 given a
# 5-transition panel and switch years spanning 2021-2025)
event_switchers = stacked[stacked['event_time'].notna()]
et_min = int(event_switchers['event_time'].min())
et_max = int(event_switchers['event_time'].max())
print(f"\nFull observed event-time span: {et_min} to {et_max} (start = earliest transition in our panel "
      f"relative to the latest switch year, end = latest transition relative to the earliest switch year)")

def bin_label(et):
    if et < 0:
        return f'et_m{abs(et)}'
    return f'et_p{et}' if et > 0 else 'et_0'

def bucket(et):
    if pd.isna(et):
        return 'control'
    return bin_label(int(et))  # no clipping -- every observed event time gets its own dummy

stacked['bucket'] = stacked['event_time'].apply(bucket)
print("\nRow counts by bucket:")
print(stacked['bucket'].value_counts().sort_index())

EVENT_TIMES = list(range(et_min, et_max + 1))
EVENT_BINS = [bin_label(e) for e in EVENT_TIMES]
for b in EVENT_BINS:
    stacked[b] = (stacked['bucket'] == b).astype(float)

stacked['baseline_rate'] = stacked['od_2020'] / (stacked['population'] / 1e5)
stacked['county_g'] = pd.Categorical(stacked['fips5']).codes

rhs_terms = ' + '.join(EVENT_BINS)

print("\n=== Raw (transition FE only), clustered by county ===")
m_raw = smf.ols(f'd_od ~ {rhs_terms} + C(transition)', data=stacked).fit(
    cov_type='cluster', cov_kwds={'groups': stacked['county_g']})
for b in EVENT_BINS:
    print(f"  {b}: b={m_raw.params[b]:+.4f} se={m_raw.bse[b]:.4f} p={m_raw.pvalues[b]:.4f}  n={int((stacked['bucket']==b).sum())}")

print("\n=== + baseline-rate control (preferred), clustered by county ===")
sub = stacked.dropna(subset=['baseline_rate'])
m_adj = smf.ols(f'd_od ~ {rhs_terms} + baseline_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})
for b in EVENT_BINS:
    print(f"  {b}: b={m_adj.params[b]:+.4f} se={m_adj.bse[b]:.4f} p={m_adj.pvalues[b]:.4f}  n={int((sub['bucket']==b).sum())}")

stacked.to_csv(os.path.join(DATA, 'eventstudy_fullspan_panel_rec.csv'), index=False)
out = dict(
    event_times=EVENT_TIMES,
    raw={b: dict(coef=float(m_raw.params[b]), se=float(m_raw.bse[b]), p=float(m_raw.pvalues[b]),
                 n=int((stacked['bucket']==b).sum())) for b in EVENT_BINS},
    adjusted={b: dict(coef=float(m_adj.params[b]), se=float(m_adj.bse[b]), p=float(m_adj.pvalues[b]),
                       n=int((sub['bucket']==b).sum())) for b in EVENT_BINS},
    n_adjusted=int(m_adj.nobs),
    cluster='county (fips5)',
)
with open(os.path.join(DATA, 'eventstudy_fullspan_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved eventstudy_fullspan_panel_rec.csv, eventstudy_fullspan_results.json")
