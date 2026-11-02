# ============================================================
# Journal of Health Economics draft -- dispensary-based alternative
# event study. Identical methodology to analyze_eventstudy_main.py
# (switcher = treatment, far-away = control, +/-3 event window,
# never-switchers omitted as reference, county-clustered SEs), but
# treatment is defined by distance to the nearest ACTUALLY OPERATING
# dispensary (real license/opening dates geocoded from state
# cannabis-regulator records, 9 of 24 legal states covered) instead
# of distance to the nearest legal-market county.
#
# Source data reused (not re-pulled): distance_by_year_dispensary.csv
# from ..\Economic Letters\New version EL\data\ (built earlier in
# this project via build_distance_dispensary.py), and the same
# overdose_county_multiyear_raw.json used everywhere else.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
SRC_DISP = os.path.join(BASE, '..', 'Economic Letters', 'New version EL', 'data')
SRC_MAIN = os.path.join(BASE, '..', 'Economic Letters', 'data')

THRESH = 50
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

dist = pd.read_csv(os.path.join(SRC_DISP, 'distance_by_year_dispensary.csv'), dtype={'fips5': str})
with open(os.path.join(SRC_MAIN, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()
dist = dist.merge(od, on='fips5', how='left')

n_total = len(dist)
dist = dist[dist['dist_2020'] >= THRESH].copy()
print(f"County universe (dispensary distances; excludes {n_total - len(dist)} always-near-at-baseline): {len(dist)}")

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
    rows.append(sub[['fips5', 'state_abbr', 'population', 'transition', 'event_time', 'd_od', 'od_2020']])

stacked = pd.concat(rows, ignore_index=True)
stacked = stacked[stacked['d_od'].notna() & (stacked['population'] > 0)].copy()

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

EVENT_BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
for b in EVENT_BINS:
    stacked[b] = (stacked['bucket'] == b).astype(float)

stacked['baseline_rate'] = stacked['od_2020'] / (stacked['population'] / 1e5)
stacked['county_g'] = pd.Categorical(stacked['fips5']).codes

rhs_terms = ' + '.join(EVENT_BINS)
sub = stacked.dropna(subset=['baseline_rate'])
m_adj = smf.ols(f'd_od ~ {rhs_terms} + baseline_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})

print("\n=== Preferred (baseline-rate-adjusted) event-study coefficients, clustered by county ===")
for b in EVENT_BINS:
    n_b = int((sub['bucket']==b).sum())
    if n_b == 0:
        print(f"  {b}: no observations")
        continue
    print(f"  {b}: b={m_adj.params[b]:+.4f} se={m_adj.bse[b]:.4f} p={m_adj.pvalues[b]:.4f}  n={n_b}")

switch0_pop = sub.loc[sub['bucket'] == 'et_0', 'population'].sum()
switch0_counties = sub.loc[sub['bucket'] == 'et_0', 'fips5'].nunique()
print(f"\nSwitch-year (et_0) bucket: {switch0_counties} counties, total population {switch0_pop:,.0f}")

stacked.to_csv(os.path.join(DATA, 'eventstudy_dispensary_panel.csv'), index=False)
out = dict(
    design='dispensary (distance to nearest actually operating dispensary)',
    adjusted={b: dict(coef=float(m_adj.params[b]), se=float(m_adj.bse[b]), p=float(m_adj.pvalues[b]),
                       n=int((sub['bucket']==b).sum())) for b in EVENT_BINS if (sub['bucket']==b).sum() > 0},
    switch0_population=float(switch0_pop),
    switch0_counties=int(switch0_counties),
    n_adjusted=int(m_adj.nobs),
    cluster='county (fips5)',
)
with open(os.path.join(DATA, 'eventstudy_dispensary_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved eventstudy_dispensary_panel.csv, eventstudy_dispensary_results.json")
