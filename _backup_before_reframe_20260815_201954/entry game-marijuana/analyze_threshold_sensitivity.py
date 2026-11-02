# ============================================================
# 70-vs-90-mile threshold sensitivity check, motivated by Nevada's
# Lincoln (79.7mi to Utah) and Elko (90.8mi to Idaho) counties sitting
# just outside the 70-mile cutoff used throughout this paper. Reruns
# the main specification for land value (all 23 states, the outcome
# where Nevada actually has data -- NASS is federal, unlike revenue
# where Nevada has no usable county series at all) and for the pooled
# revenue panels (CO+WA+OR+MT), at both thresholds.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')


def run(panel, outcome, controls, cluster_col='fips5', label=''):
    rows = []
    for thresh in [70, 90]:
        p = panel.copy()
        p['border_x'] = (p['dist_nonlegal_mi'] <= thresh).astype(int)
        p['g'] = pd.Categorical(p[cluster_col]).codes
        formula = f"{outcome} ~ border_x + {controls}"
        m = smf.ols(formula, data=p).fit(cov_type='HC1')
        mc = smf.ols(formula, data=p).fit(cov_type='cluster', cov_kwds={'groups': p['g']})
        b, se, p_hc1 = m.params['border_x'], m.bse['border_x'], m.pvalues['border_x']
        se_c, p_c = mc.bse['border_x'], mc.pvalues['border_x']
        n_border = p.loc[p['border_x'] == 1, cluster_col].nunique()
        n_states_border = p.loc[p['border_x'] == 1, 'state_abbr'].nunique()
        print(f"[{label}] {thresh}mi: b={b:+.4f} se(HC1)={se:.4f} p={p_hc1:.4f} | "
              f"se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)} "
              f"border_counties={n_border} ({n_states_border} states)")
        rows.append(dict(outcome=label, threshold=thresh, coef=b, se_hc1=se, p_hc1=p_hc1,
                          se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs), n_border_counties=n_border))
    return rows

all_rows = []

# ---------- Land value (all 23 states -- the outcome where Nevada has real data) ----------
print("=" * 70)
print("LAND VALUE -- 70mi vs 90mi threshold")
print("=" * 70)
lv = pd.read_csv(os.path.join(TABLES, 'land_value_border_panel.csv'), dtype={'fips5': str})
nv_before = lv.loc[(lv['state_abbr']=='NV') & (lv['dist_nonlegal_mi']<=70), 'fips5'].nunique()
nv_after = lv.loc[(lv['state_abbr']=='NV') & (lv['dist_nonlegal_mi']<=90), 'fips5'].nunique()
print(f"Nevada border counties: {nv_before} at 70mi -> {nv_after} at 90mi "
      f"(Lincoln 79.7mi, Elko 90.8mi both enter)")
all_rows += run(lv, 'log_landvalue', 'log_population + log_land_area + C(year) + C(state_abbr)',
                 label='land_value')

# ---------- Revenue per capita, pooled CO+WA+OR+MT ----------
print("\n" + "=" * 70)
print("REVENUE PER CAPITA (pooled CO+WA+OR+MT) -- 70mi vs 90mi threshold")
print("=" * 70)
cowaor = pd.read_csv(os.path.join(TABLES, 'pooled_cowaOR_revenue_percapita_panel.csv'), dtype={'fips5': str})
mt = pd.read_csv(os.path.join(TABLES, 'montana_revenue_percapita_panel.csv'), dtype={'fips5': str})
keep = ['fips5', 'state_abbr', 'fy_year', 'total_sales', 'population', 'dist_nonlegal_mi', 'border_70mi',
        'revenue_per_capita', 'log_revenue_per_capita', 'log_population']
pooled_rev = pd.concat([cowaor[keep], mt[keep]], ignore_index=True)
print(f"Pooled CO+WA+OR+MT: {len(pooled_rev)} obs, states {sorted(pooled_rev['state_abbr'].unique())}")
all_rows += run(pooled_rev, 'log_revenue_per_capita', 'log_population + C(fy_year) + C(state_abbr)',
                 cluster_col='fips5', label='revenue_per_capita')

# ---------- Revenue per store, pooled CO+WA+OR ----------
print("\n" + "=" * 70)
print("REVENUE PER STORE (pooled CO+WA+OR) -- 70mi vs 90mi threshold")
print("=" * 70)
ps = pd.read_csv(os.path.join(TABLES, 'pooled_cowaOR_revenue_perstore_panel.csv'), dtype={'fips5': str})
all_rows += run(ps, 'log_revenue_per_store', 'log_population + C(fy_year) + C(state_abbr)',
                 cluster_col='fips5', label='revenue_per_store')

res = pd.DataFrame(all_rows)
res.to_csv(os.path.join(TABLES, 'threshold_sensitivity_results.csv'), index=False)
print(f"\nSaved to {os.path.join(TABLES, 'threshold_sensitivity_results.csv')}")
