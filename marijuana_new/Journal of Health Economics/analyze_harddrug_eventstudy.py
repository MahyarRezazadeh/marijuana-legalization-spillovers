# ============================================================
# Journal of Health Economics draft -- hard-drug (opioid/cocaine +
# synthetic-narcotic possession) arrest event study, using the
# IDENTICAL switcher/control/event-time structure already computed
# for the overdose-mortality event study (eventstudy_main_panel.csv),
# just with a different outcome. Crosswalks FBI CDE agency-level
# pulls to county FIPS via single-county-agency name matching against
# analysis_panel.dta's county_name_p2 field (same method as
# ..\..\marijuana_substitution\analyze_partA.py).
#
# Purpose: supplementary support for the "golden policy window"
# argument in main.tex Section 4 -- tests whether hard-drug
# possession arrests show the same transitory switch-year decline
# as overdose mortality, which would support reading the mortality
# effect as a real, temporary behavioral shift.
# ============================================================
import os, re, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

def norm_county(name):
    if pd.isna(name) or not str(name).strip():
        return None
    s = str(name).upper().strip()
    s = re.sub(r"[.']", '', s)
    s = re.sub(r'\s+', ' ', s)
    for suf in [' COUNTY', ' PARISH', ' BOROUGH', ' CENSUS AREA', ' CITY AND BOROUGH',
                ' MUNICIPALITY', ' CITY']:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()

# ---- Step 1: crosswalk agency pulls to county FIPS ----
agencies = pd.read_csv(os.path.join(DATA, 'hd_agencies_raw.csv'))
arrests = pd.read_csv(os.path.join(DATA, 'hd_county_arrests_raw.csv'))
print(f"Agencies: {len(agencies)}  Arrest rows: {len(arrests)}")
print(arrests['status'].value_counts())

merged = agencies.merge(arrests, on='ori', how='inner')
ok = merged[merged['status'] == 'ok'].copy()
print(f"OK rows: {len(ok)} / {len(merged)} ({len(ok)/len(merged)*100:.1f}%)")

ok['n_counties'] = ok['counties_raw'].fillna('').apply(lambda s: len([c for c in str(s).split(',') if c.strip()]))
single = ok[ok['n_counties'] == 1].copy()
print(f"Single-county agencies: {len(single)} / {len(ok)}")
single['county_norm'] = single['counties_raw'].apply(norm_county)

panel_src = pd.read_stata(os.path.join(BASE, '..', 'data', 'analysis_panel.dta'))
panel_src['fips5'] = panel_src['fips5'].astype(str).str.zfill(5)
cw = panel_src[panel_src['year'] == 2023][['fips5', 'state_abbr', 'county_name_p2']].drop_duplicates()
cw = cw[cw['county_name_p2'].astype(str).str.strip() != '']
cw['county_norm'] = cw['county_name_p2'].apply(norm_county)
cw = cw.drop_duplicates(subset=['state_abbr', 'county_norm'])

joined = single.merge(cw[['fips5', 'state_abbr', 'county_norm']], on=['state_abbr', 'county_norm'], how='left')
matched = joined[joined['fips5'].notna()].copy()
print(f"Matched to FIPS: {len(matched)} / {len(joined)} ({len(matched)/len(joined)*100:.1f}%)")

YEARS = list(range(2020, 2026))
op_cols = [f'op_{y}' for y in YEARS]
syn_cols = [f'syn_{y}' for y in YEARS]
for c in op_cols + syn_cols:
    matched[c] = pd.to_numeric(matched[c], errors='coerce')

agg = {c: (c, 'sum') for c in op_cols + syn_cols}
county_hd = matched.groupby('fips5').agg(**agg, n_agencies=('ori', 'count')).reset_index()
for y in YEARS:
    county_hd[f'hd_{y}'] = county_hd[f'op_{y}'].fillna(0) + county_hd[f'syn_{y}'].fillna(0)
county_hd.to_csv(os.path.join(DATA, 'hd_county_arrests_matched.csv'), index=False)
print(f"Saved hd_county_arrests_matched.csv: {len(county_hd)} counties")

# ---- Step 2: merge onto the existing overdose event-study panel skeleton ----
main_panel = pd.read_csv(os.path.join(DATA, 'eventstudy_main_panel.csv'), dtype={'fips5': str})
county_hd['fips5'] = county_hd['fips5'].astype(str).str.zfill(5)

coverage = set(main_panel['fips5'].unique()) & set(county_hd['fips5'])
print(f"Coverage: {len(coverage)} / {main_panel['fips5'].nunique()} event-study counties have arrest data")

hd_cols = ['fips5'] + [f'hd_{y}' for y in YEARS]
s = main_panel.merge(county_hd[hd_cols], on='fips5', how='inner')

def year_pre_post(transition):
    y0, y1 = transition.split('-')
    return int(y0), int(y1)

rows = []
for _, r in s.iterrows():
    y0, y1 = year_pre_post(r['transition'])
    hd_pre, hd_post = r[f'hd_{y0}'], r[f'hd_{y1}']
    if pd.isna(hd_pre) or pd.isna(hd_post):
        continue
    rate_pre = hd_pre / (r['population'] / 1e5)
    rate_post = hd_post / (r['population'] / 1e5)
    rows.append({
        'fips5': r['fips5'], 'state_abbr': r['state_abbr'], 'population': r['population'],
        'transition': r['transition'], 'bucket': r['bucket'],
        'd_hd': rate_post - rate_pre, 'hd_2020': r['hd_2020'],
    })

hd_panel = pd.DataFrame(rows)
hd_panel['baseline_hd_rate'] = hd_panel['hd_2020'] / (hd_panel['population'] / 1e5)
hd_panel['county_g'] = pd.Categorical(hd_panel['fips5']).codes
print(f"\nHard-drug event-study panel: {len(hd_panel)} rows")
print(hd_panel['bucket'].value_counts().sort_index())

EVENT_BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
for b in EVENT_BINS:
    hd_panel[b] = (hd_panel['bucket'] == b).astype(float)

sub = hd_panel.dropna(subset=['baseline_hd_rate'])
rhs = ' + '.join(EVENT_BINS)
m = smf.ols(f'd_hd ~ {rhs} + baseline_hd_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})

print("\n=== Hard-drug possession arrest event study (baseline-rate-adjusted, county-clustered) ===")
out = {}
for b in EVENT_BINS:
    n_b = int((sub['bucket'] == b).sum())
    if n_b == 0:
        continue
    out[b] = dict(coef=float(m.params[b]), se=float(m.bse[b]), p=float(m.pvalues[b]), n=n_b)
    print(f"  {b}: b={m.params[b]:+.4f} se={m.bse[b]:.4f} p={m.pvalues[b]:.4f}  n={n_b}")

hd_panel.to_csv(os.path.join(DATA, 'eventstudy_harddrug_panel.csv'), index=False)
with open(os.path.join(DATA, 'eventstudy_harddrug_results.json'), 'w') as f:
    json.dump(dict(adjusted=out, n=int(m.nobs), cluster='county (fips5)'), f, indent=2)
print("\nSaved eventstudy_harddrug_panel.csv, eventstudy_harddrug_results.json")
