# ============================================================
# Pre-trends / event-study check on the recreational stacked
# design: tests whether counties that eventually switch already
# show a divergent overdose trend in the transition(s) BEFORE they
# switch, relative to counties that never switch in the window.
# This is the natural follow-up flagged as Limitation (4) in the
# Economic Inquiry draft -- the per-transition breakdown already in
# the paper (Figure 3) shows effects AFTER switching are not driven
# by one period, but says nothing about anticipation/pre-trends.
#
# Design: event-time dummies relative to each eventual-switcher
# county's own switch year, estimated against NEVER-switching
# counties as the omitted reference group, with transition fixed
# effects and state-clustered SE -- a standard dynamic/event-study
# specification (Callaway-Sant'Anna-style not-yet-treated control).
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

THRESH = 50
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

dist = pd.read_csv(os.path.join(DATA, 'distance_by_year_rec.csv'), dtype={'fips5': str})
print(f"County universe (not always-near): {len(dist)}")

# ── identify each county's switch year (first year dist<50 while prior year dist>=50) ──
def find_switch_year(row):
    for y in YEARS[1:]:
        if row[f'dist_{y-1}'] >= THRESH and row[f'dist_{y}'] < THRESH:
            return y
    return np.nan

dist['switch_year'] = dist.apply(find_switch_year, axis=1)
n_switchers = dist['switch_year'].notna().sum()
print(f"Eventual switchers: {n_switchers}, never-switchers: {len(dist) - n_switchers}")
print(dist['switch_year'].value_counts().sort_index())

# ── build stacked transition rows with event-time tags ──
with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()
dist = dist.merge(od, on='fips5', how='left')

rows = []
for t in YEARS[1:]:
    t0 = t - 1
    # a county is eligible in transition t if it hasn't switched in an earlier transition
    eligible = dist[(dist['switch_year'].isna()) | (dist['switch_year'] >= t)].copy()
    eligible['transition'] = f'{t0}-{t}'
    eligible['event_time'] = t - eligible['switch_year']  # 0 = switch transition, -1 = one transition before, etc.
    eligible['od_rate_pre'] = eligible[f'od_{t0}'] / (eligible['population'] / 1e5) if f'od_{t0}' in eligible else np.nan
    eligible['od_rate_post'] = eligible[f'od_{t}'] / (eligible['population'] / 1e5) if f'od_{t}' in eligible else np.nan
    eligible['d_od'] = eligible['od_rate_post'] - eligible['od_rate_pre']
    rows.append(eligible[['fips5', 'state_abbr', 'population', 'transition', 'event_time', 'd_od']])

stacked = pd.concat(rows, ignore_index=True)
stacked = stacked[stacked['d_od'].notna() & (stacked['population'] > 0)].copy()

# bucket event time: lead2plus (<=-2), lead1 (-1), switch0 (0). Never-switchers have NaN event_time -> reference.
def bucket(et):
    if pd.isna(et):
        return 'control'
    if et <= -2:
        return 'lead2plus'
    if et == -1:
        return 'lead1'
    if et == 0:
        return 'switch0'
    return 'drop'  # shouldn't occur (post-switch rows are excluded from the panel already)

stacked['bucket'] = stacked['event_time'].apply(bucket)
print("\nRow counts by bucket:")
print(stacked['bucket'].value_counts())

stacked = stacked[stacked['bucket'] != 'drop'].copy()
for b in ['lead1', 'lead2plus', 'switch0']:
    stacked[b] = (stacked['bucket'] == b).astype(float)

stacked['g'] = pd.Categorical(stacked['state_abbr']).codes
model = smf.ols('d_od ~ lead2plus + lead1 + switch0 + C(transition)', data=stacked).fit(
    cov_type='cluster', cov_kwds={'groups': stacked['g']})

print("\n=== Event-study (never-switchers = reference group) ===")
for term in ['lead2plus', 'lead1', 'switch0']:
    print(f"{term:12s} b={model.params[term]:+.4f} se={model.bse[term]:.4f} p={model.pvalues[term]:.4f} "
          f"n={(stacked['bucket']==term).sum()}")

f_pretrend = model.f_test('lead1 = 0, lead2plus = 0')
print(f"\nJoint F-test (both leads = 0): F={float(f_pretrend.fvalue):.4f} p={float(f_pretrend.pvalue):.4f}")

out = {
    term: dict(coef=float(model.params[term]), se=float(model.bse[term]), p=float(model.pvalues[term]),
               n=int((stacked['bucket'] == term).sum()))
    for term in ['lead2plus', 'lead1', 'switch0']
}
out['n_control'] = int((stacked['bucket'] == 'control').sum())
out['joint_f_leads'] = dict(F=float(f_pretrend.fvalue), p=float(f_pretrend.pvalue))
with open(os.path.join(DATA, 'pretrends_rec.json'), 'w') as f:
    json.dump(out, f, indent=2)

pd.DataFrame([
    dict(term='2+ transitions before switching', **out['lead2plus']),
    dict(term='1 transition before switching', **out['lead1']),
    dict(term='Switch transition (main effect)', **out['switch0']),
]).to_csv(os.path.join(TABLES, 'pretrends_rec.csv'), index=False)
stacked.to_csv(os.path.join(DATA, 'pretrends_panel_rec.csv'), index=False)
print("\nSaved pretrends_rec.json, tables/pretrends_rec.csv, pretrends_panel_rec.csv")
