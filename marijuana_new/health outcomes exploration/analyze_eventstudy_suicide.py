# ============================================================
# Exploratory outcome swap on the main paper's own design: does the
# switch-year effect on overdose mortality (the paper's headline
# finding) also show up for SUICIDE mortality? Same treatment
# (distance to nearest legal-market county crosses 50mi), same
# stacked-transition/event-time logic, same never-switcher control
# group, same baseline-rate mean-reversion control -- only the
# outcome and its source data change.
#
# Outcome source: CDC "Mapping Injury, Overdose, and Violence -
# County" (data.cdc.gov resource psx4-wq38), intent=All_Suicide,
# which reports a pre-computed rate per 100,000 directly (no need to
# reconstruct from suppressed small-count ranges the way raw counts
# would require). Covers discrete calendar years 2019-2024; since the
# treatment/distance panel (../Economic Letters/data/distance_by_year_
# rec_full.csv) only has dist_2020 through dist_2025, and this outcome
# has no clean full-year 2025 entry yet (only a rolling trailing-
# twelve-month figure), the panel here runs one transition SHORTER
# than the main design: 2020-2021 through 2023-2024 (4 transitions,
# vs. the main design's 5).
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
SRC = os.path.join(BASE, '..', 'Economic Letters', 'data')

THRESH = 50
YEARS = [2020, 2021, 2022, 2023, 2024]  # one shorter than main design (no clean 2025 suicide data yet)

dist = pd.read_csv(os.path.join(SRC, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})

with open(os.path.join(DATA, 'raw', 'cdc_suicide_all_counties.json'), encoding='utf-8') as f:
    raw = pd.DataFrame(json.load(f))
raw = raw[raw['period'].isin([str(y) for y in YEARS])].copy()
raw['fips5'] = raw['geoid'].astype(str).str.zfill(5)
raw['year'] = raw['period'].astype(int)
raw['rate'] = pd.to_numeric(raw['rate'], errors='coerce')
raw.loc[raw['rate'] == -999, 'rate'] = np.nan  # CDC sentinel for suppressed/unreliable rate

su = raw.pivot_table(index='fips5', columns='year', values='rate', aggfunc='first')
su.columns = [f'su_{c}' for c in su.columns]
su = su.reset_index()
dist = dist.merge(su, on='fips5', how='left')

dist = dist[dist['dist_2020'] >= THRESH].copy()
print(f"County universe (excludes always-near-at-baseline): {len(dist)}")

def find_switch_year(row):
    for y in YEARS[1:]:
        if row[f'dist_{y-1}'] >= THRESH and row[f'dist_{y}'] < THRESH:
            return y
    return np.nan

dist['switch_year'] = dist.apply(find_switch_year, axis=1)
n_switchers = dist['switch_year'].notna().sum()
print(f"Eventual switchers (treatment): {n_switchers}, never-switchers (far-away control): {len(dist) - n_switchers}")

rows = []
for t in YEARS[1:]:
    t0 = t - 1
    sub = dist.copy()
    sub['transition'] = f'{t0}-{t}'
    sub['event_time'] = t - sub['switch_year']
    sub['d_su'] = sub[f'su_{t}'] - sub[f'su_{t0}']
    rows.append(sub[['fips5', 'state_abbr', 'population', 'transition', 'event_time', 'd_su', 'su_2020']])

stacked = pd.concat(rows, ignore_index=True)
stacked = stacked[stacked['d_su'].notna() & (stacked['population'] > 0)].copy()

def bin_label(et):
    if et < 0:
        return f'et_m{abs(et)}'
    return f'et_p{et}' if et > 0 else 'et_0'

def bucket(et):
    if pd.isna(et):
        return 'control'
    return bin_label(max(-3, min(3, int(et))))

stacked['bucket'] = stacked['event_time'].apply(bucket)
print("\nRow counts by bucket:")
print(stacked['bucket'].value_counts().sort_index())

EVENT_BINS = [b for b in ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
              if b in stacked['bucket'].unique()]
for b in EVENT_BINS:
    stacked[b] = (stacked['bucket'] == b).astype(float)

stacked['baseline_rate'] = stacked['su_2020']
stacked['county_g'] = pd.Categorical(stacked['fips5']).codes

rhs_terms = ' + '.join(EVENT_BINS)
sub = stacked.dropna(subset=['baseline_rate'])
m_adj = smf.ols(f'd_su ~ {rhs_terms} + baseline_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})

print("\n=== Event-study coefficients (outcome: change in suicide rate per 100k), county-clustered ===")
for b in EVENT_BINS:
    print(f"  {b}: b={m_adj.params[b]:+.4f} se={m_adj.bse[b]:.4f} p={m_adj.pvalues[b]:.4f}  n={int((sub['bucket']==b).sum())}")

stacked.to_csv(os.path.join(DATA, 'eventstudy_suicide_panel.csv'), index=False)
out = dict(
    design='suicide mortality (same distance-to-legal-market treatment as main overdose design)',
    adjusted={b: dict(coef=float(m_adj.params[b]), se=float(m_adj.bse[b]), p=float(m_adj.pvalues[b]),
                       n=int((sub['bucket'] == b).sum())) for b in EVENT_BINS},
    n_adjusted=int(m_adj.nobs),
    cluster='county (fips5)',
)
with open(os.path.join(DATA, 'eventstudy_suicide_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved eventstudy_suicide_panel.csv, eventstudy_suicide_results.json")
