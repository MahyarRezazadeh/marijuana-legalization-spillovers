# ============================================================
# Robustness check requested 2026-08-06: drop the COVID years
# (2020, 2021) from the main overdose-mortality event study and
# re-run. Mirrors analyze_eventstudy_main.py exactly (same design:
# switcher = crosses 50mi distance-to-nearest-legal-market-county
# threshold; control = far-away never-switchers; county-clustered
# SEs), except:
#   - YEARS starts at 2022 instead of 2020, so transitions are
#     2022-2023, 2023-2024, 2024-2025 only (2020-2021, 2021-2022
#     transitions dropped entirely -- both endpoints touch a COVID
#     year for the latter).
#   - The "always-near-at-baseline" universe filter and the
#     baseline_rate mean-reversion control both re-anchor on 2022
#     (the new first observed year) instead of 2020.
# Outputs are saved under _nocovid suffixes so the original
# 2020-2025 results are untouched.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
SRC = os.path.join(BASE, '..', 'Economic Letters', 'data')

THRESH = 50
YEARS = [2022, 2023, 2024, 2025]
BASELINE_YEAR = 2022

dist = pd.read_csv(os.path.join(SRC, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
with open(os.path.join(SRC, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()
dist = dist.merge(od, on='fips5', how='left')

dist = dist[dist[f'dist_{BASELINE_YEAR}'] >= THRESH].copy()
print(f"County universe (excludes {2331 - len(dist)} always-near-as-of-{BASELINE_YEAR}): {len(dist)}")

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
    sub['od_rate_pre'] = sub[f'od_{t0}'] / (sub['population'] / 1e5)
    sub['od_rate_post'] = sub[f'od_{t}'] / (sub['population'] / 1e5)
    sub['d_od'] = sub['od_rate_post'] - sub['od_rate_pre']
    rows.append(sub[['fips5', 'state_abbr', 'population', 'transition', 'event_time', 'd_od', f'od_{BASELINE_YEAR}']])

stacked = pd.concat(rows, ignore_index=True)
stacked = stacked.rename(columns={f'od_{BASELINE_YEAR}': 'od_baseline'})
stacked = stacked[stacked['d_od'].notna() & (stacked['population'] > 0)].copy()

def bin_label(et):
    if et < 0:
        return f'et_m{abs(et)}'
    return f'et_p{et}' if et > 0 else 'et_0'

def bucket(et):
    if pd.isna(et):
        return 'control'
    return bin_label(max(-1, min(2, int(et))))

stacked['bucket'] = stacked['event_time'].apply(bucket)
print("\nRow counts by bucket:")
print(stacked['bucket'].value_counts().sort_index())

# Note: et_m2 dropped -- no county's switch year fell in 2025 within
# this trimmed 2022-2025 window (retail legalization mostly happened
# earlier), so that bin has zero supporting observations.
EVENT_BINS = ['et_m1', 'et_0', 'et_p1', 'et_p2']
for b in EVENT_BINS:
    stacked[b] = (stacked['bucket'] == b).astype(float)

stacked['baseline_rate'] = stacked['od_baseline'] / (stacked['population'] / 1e5)
stacked['county_g'] = pd.Categorical(stacked['fips5']).codes

rhs_terms = ' + '.join(EVENT_BINS)
sub = stacked.dropna(subset=['baseline_rate'])
m_adj = smf.ols(f'd_od ~ {rhs_terms} + baseline_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})

print("\n=== No-COVID (2022-2025 only), baseline-rate-adjusted event-study coefficients, clustered by county ===")
for b in EVENT_BINS:
    print(f"  {b}: b={m_adj.params[b]:+.4f} se={m_adj.bse[b]:.4f} p={m_adj.pvalues[b]:.4f}  n={int((sub['bucket']==b).sum())}")

switch0_pop = sub.loc[sub['bucket'] == 'et_0', 'population'].sum()
switch0_counties = sub.loc[sub['bucket'] == 'et_0', 'fips5'].nunique()
print(f"\nSwitch-year (et_0) bucket: {switch0_counties} counties, total population {switch0_pop:,.0f}")

stacked.to_csv(os.path.join(DATA, 'eventstudy_main_nocovid_panel.csv'), index=False)
out = dict(
    design='main (distance to nearest legal-market county), COVID years 2020-2021 dropped',
    years_used=YEARS,
    adjusted={b: dict(coef=float(m_adj.params[b]), se=float(m_adj.bse[b]), p=float(m_adj.pvalues[b]),
                       n=int((sub['bucket']==b).sum())) for b in EVENT_BINS},
    switch0_population=float(switch0_pop),
    switch0_counties=int(switch0_counties),
    n_adjusted=int(m_adj.nobs),
    cluster='county (fips5)',
)
with open(os.path.join(DATA, 'eventstudy_main_nocovid_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved eventstudy_main_nocovid_panel.csv, eventstudy_main_nocovid_results.json")
