# ============================================================
# Journal of Health Economics draft -- threshold robustness. Reruns
# the same +/-3-year event-study specification (analyze_eventstudy_
# main.py / analyze_eventstudy_dispensary.py) at 75-mile and
# 100-mile switcher thresholds, for both the main (legal-market-
# county distance) and alternative (operating-dispensary distance)
# designs. 50 miles remains the paper's primary threshold (already
# built); this script only adds 75 and 100 as robustness checks.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
SRC_MAIN = os.path.join(BASE, '..', 'Economic Letters', 'data')
SRC_DISP = os.path.join(BASE, '..', 'Economic Letters', 'New version EL', 'data')

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

with open(os.path.join(SRC_MAIN, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()

def bin_label(et):
    if et < 0:
        return f'et_m{abs(et)}'
    return f'et_p{et}' if et > 0 else 'et_0'

EVENT_BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']

def run_eventstudy(design, thresh, dist_path):
    dist = pd.read_csv(dist_path, dtype={'fips5': str})
    dist = dist.merge(od, on='fips5', how='left')
    n_total = len(dist)
    dist = dist[dist['dist_2020'] >= thresh].copy()

    def find_switch_year(row):
        for y in YEARS[1:]:
            if row[f'dist_{y-1}'] >= thresh and row[f'dist_{y}'] < thresh:
                return y
        return np.nan

    dist['switch_year'] = dist.apply(find_switch_year, axis=1)
    n_switchers = int(dist['switch_year'].notna().sum())
    n_control = len(dist) - n_switchers

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

    def bucket(et):
        if pd.isna(et):
            return 'control'
        return bin_label(max(-3, min(3, int(et))))

    stacked['bucket'] = stacked['event_time'].apply(bucket)
    for b in EVENT_BINS:
        stacked[b] = (stacked['bucket'] == b).astype(float)
    stacked['baseline_rate'] = stacked['od_2020'] / (stacked['population'] / 1e5)
    stacked['county_g'] = pd.Categorical(stacked['fips5']).codes

    sub = stacked.dropna(subset=['baseline_rate'])
    rhs_terms = ' + '.join(EVENT_BINS)
    m_adj = smf.ols(f'd_od ~ {rhs_terms} + baseline_rate + C(transition)', data=sub).fit(
        cov_type='cluster', cov_kwds={'groups': sub['county_g']})

    print(f"\n=== {design}, threshold={thresh}mi ===")
    print(f"  Universe: {n_total} counties (excludes always-near); switchers={n_switchers}, controls={n_control}")
    adj_out = {}
    for b in EVENT_BINS:
        n_b = int((sub['bucket'] == b).sum())
        if n_b == 0:
            continue
        adj_out[b] = dict(coef=float(m_adj.params[b]), se=float(m_adj.bse[b]), p=float(m_adj.pvalues[b]), n=n_b)
        print(f"  {b}: b={m_adj.params[b]:+.4f} se={m_adj.bse[b]:.4f} p={m_adj.pvalues[b]:.4f}  n={n_b}")

    switch0_pop = sub.loc[sub['bucket'] == 'et_0', 'population'].sum()
    switch0_counties = sub.loc[sub['bucket'] == 'et_0', 'fips5'].nunique()

    stacked.to_csv(os.path.join(DATA, f'eventstudy_{design}_t{thresh}_panel.csv'), index=False)
    out = dict(design=f'{design} (threshold={thresh}mi)', threshold=thresh, n_switchers=n_switchers,
               n_control=n_control, adjusted=adj_out, switch0_population=float(switch0_pop),
               switch0_counties=int(switch0_counties), n_adjusted=int(m_adj.nobs), cluster='county (fips5)')
    with open(os.path.join(DATA, f'eventstudy_{design}_t{thresh}_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return out

if __name__ == '__main__':
    for thresh in [75, 100]:
        run_eventstudy('main', thresh, os.path.join(SRC_MAIN, 'distance_by_year_rec_full.csv'))
        run_eventstudy('dispensary', thresh, os.path.join(SRC_DISP, 'distance_by_year_dispensary.csv'))
    print("\nDone. Saved eventstudy_{main,dispensary}_t{75,100}_panel.csv / _results.json")
