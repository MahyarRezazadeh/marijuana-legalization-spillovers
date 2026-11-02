# ============================================================
# Runs the pooled stacked-transition regressions (switcher binary
# + continuous per-mile) on both the recreational (primary) and
# medical (robustness) designs, plus a per-transition event-study
# breakdown for the recreational design.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
os.makedirs(TABLES, exist_ok=True)

with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()

def ols_cl(y, X, groups, xname):
    ok = y.notna() & X[xname].notna() & groups.notna()
    y2, X2, g2 = y[ok], X[ok], groups[ok]
    gcodes = pd.Categorical(g2).codes
    Xc = sm.add_constant(X2)
    res = sm.OLS(y2, Xc).fit(cov_type='cluster', cov_kwds={'groups': gcodes})
    return dict(coef=float(res.params[xname]), se=float(res.bse[xname]),
                p=float(res.pvalues[xname]), n=int(res.nobs))

def build_transition_outcome(stacked):
    stacked = stacked.merge(od, on='fips5', how='left')
    stacked['population'] = pd.to_numeric(stacked['population'], errors='coerce')
    rate_pre = stacked.apply(lambda r: r.get(f"od_{int(r['year_pre'])}", np.nan), axis=1)
    rate_post = stacked.apply(lambda r: r.get(f"od_{int(r['year_post'])}", np.nan), axis=1)
    stacked['od_rate_pre'] = pd.to_numeric(rate_pre, errors='coerce') / (stacked['population'] / 1e5)
    stacked['od_rate_post'] = pd.to_numeric(rate_post, errors='coerce') / (stacked['population'] / 1e5)
    stacked['d_od'] = stacked['od_rate_post'] - stacked['od_rate_pre']
    return stacked

def run_design(label):
    stacked = pd.read_csv(os.path.join(DATA, f'stacked_panel_{label}.csv'), dtype={'fips5': str})
    stacked = build_transition_outcome(stacked)
    stacked = stacked[stacked['population'] > 0].copy()

    trans_dummies = pd.get_dummies(stacked['transition'], prefix='t', drop_first=True).astype(float)
    Xsw = pd.concat([stacked[['switcher']], trans_dummies], axis=1)
    Xdist = pd.concat([stacked[['d_dist_mi']], trans_dummies], axis=1)

    r_sw = ols_cl(stacked['d_od'], Xsw, stacked['state_abbr'], 'switcher')
    r_dist = ols_cl(stacked['d_od'], Xdist, stacked['state_abbr'], 'd_dist_mi')

    avail = stacked[stacked['d_od'].notna()]
    n_sw_events = int(avail[avail['switcher'] == 1].shape[0])
    n_ct_obs = int(avail[avail['switcher'] == 0].shape[0])
    print(f"\n=== {label} design (pooled, transition FE, state-clustered SE) ===")
    print(f"N transition-rows with outcome: {len(avail)} ({n_sw_events} switch events, {n_ct_obs} control-period obs)")
    print(f"switcher:  b={r_sw['coef']:+.4f} se={r_sw['se']:.4f} p={r_sw['p']:.4f} n={r_sw['n']}")
    print(f"dist_mile: b={r_dist['coef']:+.5f} se={r_dist['se']:.5f} p={r_dist['p']:.4f} n={r_dist['n']}")

    per_transition = []
    for t, sub in stacked.groupby('transition'):
        subavail = sub[sub['d_od'].notna()]
        if subavail['switcher'].sum() < 3 or (subavail['switcher'] == 0).sum() < 3:
            continue
        rr = ols_cl(sub['d_od'], sub[['switcher']], sub['state_abbr'], 'switcher')
        per_transition.append(dict(transition=t, **rr,
                                    n_switchers=int(subavail['switcher'].sum()),
                                    n_controls=int((subavail['switcher'] == 0).sum())))
    pt = pd.DataFrame(per_transition).sort_values('transition')
    print(pt.to_string(index=False))

    out = dict(switcher=r_sw, dist_mile=r_dist,
               n_switch_events=n_sw_events, n_control_obs=n_ct_obs)
    with open(os.path.join(DATA, f'multiperiod_results_{label}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    pt.to_csv(os.path.join(TABLES, f'event_study_{label}.csv'), index=False)
    stacked.to_csv(os.path.join(DATA, f'stacked_panel_{label}_full.csv'), index=False)

    rows = [dict(spec='switcher', **r_sw), dict(spec='dist_mile', **r_dist)]
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, f'tE1_multiperiod_{label}.csv'), index=False)
    return out

rec_out = run_design('rec')
med_out = run_design('med')

print("\nSaved multiperiod_results_{rec,med}.json, tE1_multiperiod_{rec,med}.csv, event_study_{rec,med}.csv")
