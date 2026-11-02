# ============================================================
# Generalizes the pooled main specification (Table 1) to let both
# the switcher effect and the distance effect vary freely by
# transition period, via full interaction terms, instead of
# assuming one pooled coefficient across all five transitions.
#
#   d_od_it = a + sum_t [g_t * (Switcher_it x Period_t)]
#                + sum_t [b_t * (Dist_it x Period_t)]
#                + d_t (period FE) + l*baseline_rate_i + e_it
#
# Unit of observation: (county i, transition t), the same
# county-transition panel behind Table 1/2 (stacked_panel_rec_
# full.csv) -- i.e. one row per county per one-year transition it
# is "at risk" of switching in (dropped from later transitions
# once it switches, per the main design).
#
# This nests the pooled specification: imposing g_t=g and b_t=b
# for all t recovers Table 1 exactly. We report the period-
# specific coefficients and an F-test of poolability (are all g_t
# equal to each other, and all b_t equal to each other).
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)
s = s.dropna(subset=['baseline_rate'])
s['g'] = pd.Categorical(s['state_abbr']).codes

TRANSITIONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
s['transition'] = pd.Categorical(s['transition'], categories=TRANSITIONS, ordered=True)

# build period FE plus fully separate per-period switcher/distance
# coefficients directly (one dummy-interaction column per transition,
# per variable) rather than via patsy's C(transition):switcher syntax,
# for an unambiguous one-coefficient-per-period-per-variable mapping.
# Column names use t1..t5 (not the literal "2020-2021" strings) since
# patsy's formula parser chokes on hyphens/digits in identifiers.
#
# 2020-2021 and 2024-2025 have ZERO switch events between them after
# the baseline-rate restriction (all 177 switch events fall in the
# middle three transitions), so a "switcher effect for 2020-2021"
# column would be all zeros -- exactly the degenerate, rank-deficient
# case that the original per-transition event study (Table A2)
# already excludes those two transitions from for the same reason. We
# do the same here: those two transitions still contribute their rows
# to the period fixed effects and the baseline-rate estimate, but do
# not get their own separate switcher/distance coefficient.
VIABLE = ['2021-2022', '2022-2023', '2023-2024']
T_LABEL = {f't{i+1}': t for i, t in enumerate(VIABLE)}
for key, t in T_LABEL.items():
    s[f'sw_{key}'] = ((s['transition'] == t) * s['switcher']).astype(float)
    s[f'dist_{key}'] = ((s['transition'] == t) * s['d_dist_mi']).astype(float)

rhs_terms = ['C(transition)'] + [f'sw_{k}' for k in T_LABEL] + [f'dist_{k}' for k in T_LABEL] + ['baseline_rate']
formula = 'd_od ~ ' + ' + '.join(rhs_terms)
m = smf.ols(formula, data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g']})

print("=== Period-specific switcher and distance coefficients ===")
rows = []
for key, t in T_LABEL.items():
    sw_term, dist_term = f'sw_{key}', f'dist_{key}'
    n_sw = int(s.loc[(s['transition'] == t) & (s['switcher'] == 1)].shape[0])
    row = dict(transition=t,
               sw_coef=m.params.get(sw_term, np.nan), sw_se=m.bse.get(sw_term, np.nan), sw_p=m.pvalues.get(sw_term, np.nan),
               dist_coef=m.params.get(dist_term, np.nan), dist_se=m.bse.get(dist_term, np.nan), dist_p=m.pvalues.get(dist_term, np.nan),
               n_switchers=n_sw)
    rows.append(row)
    print(f"{t}: switcher b={row['sw_coef']:+.4f} se={row['sw_se']:.4f} p={row['sw_p']:.4f} (n_sw={n_sw}) | "
          f"dist b={row['dist_coef']:+.5f} se={row['dist_se']:.5f} p={row['dist_p']:.4f}")

pool_sw = m.f_test(' = '.join([f'sw_{k}' for k in T_LABEL]))
pool_dist = m.f_test(' = '.join([f'dist_{k}' for k in T_LABEL]))
print(f"\nF-test: all switcher period-coefficients equal (poolable): F={float(pool_sw.fvalue):.4f} p={float(pool_sw.pvalue):.4f}")
print(f"F-test: all distance period-coefficients equal (poolable): F={float(pool_dist.fvalue):.4f} p={float(pool_dist.pvalue):.4f}")

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(TABLES, 'period_interactions.csv'), index=False)
out = dict(by_period=rows,
           poolability_switcher=dict(F=float(pool_sw.fvalue), p=float(pool_sw.pvalue)),
           poolability_distance=dict(F=float(pool_dist.fvalue), p=float(pool_dist.pvalue)),
           n=int(m.nobs))
with open(os.path.join(DATA, 'period_interactions.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nN={int(m.nobs)}. Saved period_interactions.csv, period_interactions.json")
