# ============================================================
# Generalizes the pooled main specification (Table 1) to let both
# the switcher effect and the distance effect vary freely by
# transition period, via full interaction terms, instead of
# assuming one pooled coefficient across all five transitions.
#
# Adapted for "New version EL": reads the local 50-mile stacked
# panel (data/stacked_panel_main_t50.csv, already carries a
# baseline_rate column) instead of the parent folder's
# stacked_panel_rec_full.csv -- same underlying design/values,
# confirmed by matching point estimates. Adds a second,
# county-clustered SE alongside the original state-clustered one
# for the county-clustered-SE-column request.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_main_t50.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s = s.dropna(subset=['baseline_rate'])
s['g'] = pd.Categorical(s['state_abbr']).codes
s['g_county'] = pd.Categorical(s['fips5']).codes

TRANSITIONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
s['transition'] = pd.Categorical(s['transition'], categories=TRANSITIONS, ordered=True)

VIABLE = ['2021-2022', '2022-2023', '2023-2024']
T_LABEL = {f't{i+1}': t for i, t in enumerate(VIABLE)}
for key, t in T_LABEL.items():
    s[f'sw_{key}'] = ((s['transition'] == t) * s['switcher']).astype(float)
    s[f'dist_{key}'] = ((s['transition'] == t) * s['d_dist_mi']).astype(float)

rhs_terms = ['C(transition)'] + [f'sw_{k}' for k in T_LABEL] + [f'dist_{k}' for k in T_LABEL] + ['baseline_rate']
formula = 'd_od ~ ' + ' + '.join(rhs_terms)
m = smf.ols(formula, data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g']})
m_c = smf.ols(formula, data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g_county']})

print("=== Period-specific switcher and distance coefficients (state- vs county-clustered SE) ===")
rows = []
for key, t in T_LABEL.items():
    sw_term, dist_term = f'sw_{key}', f'dist_{key}'
    n_sw = int(s.loc[(s['transition'] == t) & (s['switcher'] == 1)].shape[0])
    row = dict(transition=t,
               sw_coef=m.params.get(sw_term, np.nan), sw_se=m.bse.get(sw_term, np.nan), sw_p=m.pvalues.get(sw_term, np.nan),
               sw_se_county=m_c.bse.get(sw_term, np.nan), sw_p_county=m_c.pvalues.get(sw_term, np.nan),
               dist_coef=m.params.get(dist_term, np.nan), dist_se=m.bse.get(dist_term, np.nan), dist_p=m.pvalues.get(dist_term, np.nan),
               dist_se_county=m_c.bse.get(dist_term, np.nan), dist_p_county=m_c.pvalues.get(dist_term, np.nan),
               n_switchers=n_sw)
    rows.append(row)
    print(f"{t}: switcher b={row['sw_coef']:+.4f} se={row['sw_se']:.4f} p={row['sw_p']:.4f} "
          f"| se_county={row['sw_se_county']:.4f} p_county={row['sw_p_county']:.4f} (n_sw={n_sw})")
    print(f"       dist b={row['dist_coef']:+.5f} se={row['dist_se']:.5f} p={row['dist_p']:.4f} "
          f"| se_county={row['dist_se_county']:.5f} p_county={row['dist_p_county']:.4f}")

pool_sw = m.f_test(' = '.join([f'sw_{k}' for k in T_LABEL]))
pool_dist = m.f_test(' = '.join([f'dist_{k}' for k in T_LABEL]))
pool_sw_c = m_c.f_test(' = '.join([f'sw_{k}' for k in T_LABEL]))
pool_dist_c = m_c.f_test(' = '.join([f'dist_{k}' for k in T_LABEL]))
print(f"\nF-test (state-clustered): switcher period-coeffs equal: F={float(pool_sw.fvalue):.4f} p={float(pool_sw.pvalue):.4f}")
print(f"F-test (state-clustered): distance period-coeffs equal: F={float(pool_dist.fvalue):.4f} p={float(pool_dist.pvalue):.4f}")
print(f"F-test (county-clustered): switcher period-coeffs equal: F={float(pool_sw_c.fvalue):.4f} p={float(pool_sw_c.pvalue):.4f}")
print(f"F-test (county-clustered): distance period-coeffs equal: F={float(pool_dist_c.fvalue):.4f} p={float(pool_dist_c.pvalue):.4f}")

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(TABLES, 'period_interactions.csv'), index=False)
out = dict(by_period=rows,
           poolability_switcher=dict(F=float(pool_sw.fvalue), p=float(pool_sw.pvalue)),
           poolability_distance=dict(F=float(pool_dist.fvalue), p=float(pool_dist.pvalue)),
           poolability_switcher_county=dict(F=float(pool_sw_c.fvalue), p=float(pool_sw_c.pvalue)),
           poolability_distance_county=dict(F=float(pool_dist_c.fvalue), p=float(pool_dist_c.pvalue)),
           n=int(m.nobs))
with open(os.path.join(DATA, 'period_interactions.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nN={int(m.nobs)}. Saved period_interactions.csv, period_interactions.json")
