import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel_new.csv'))
main = panel[panel['d_od'].notna()].copy()
print(f"N = {len(main)}")

def cluster_fit(formula, df):
    mod = smf.ols(formula, data=df)
    g = pd.Categorical(df['state_abbr']).codes
    return mod.fit(cov_type='cluster', cov_kwds={'groups': g})

res_lin = cluster_fit('d_od ~ d_dist_mi', main)
main['dist2'] = main['d_dist_mi'] ** 2
res_quad = cluster_fit('d_od ~ d_dist_mi + dist2', main)
main['dist3'] = main['d_dist_mi'] ** 3
res_cubic = cluster_fit('d_od ~ d_dist_mi + dist2 + dist3', main)

print("\n=== LINEAR ==="); print(res_lin.summary().tables[1])
print("\n=== QUADRATIC ==="); print(res_quad.summary().tables[1])
print(f"Quadratic term p-value: {res_quad.pvalues['dist2']:.4f}")
f_test_quad = res_quad.f_test('dist2 = 0')
f_test_cubic = res_cubic.f_test('dist2 = 0, dist3 = 0')
print(f"F-test (quadratic=0): p={float(f_test_quad.pvalue):.4f}")
print(f"F-test (quad+cubic=0): p={float(f_test_cubic.pvalue):.4f}")

out = {
    'lin_b1': res_lin.params['d_dist_mi'], 'lin_se1': res_lin.bse['d_dist_mi'],
    'quad_b1': res_quad.params['d_dist_mi'], 'quad_se1': res_quad.bse['d_dist_mi'],
    'quad_b2': res_quad.params['dist2'], 'quad_se2': res_quad.bse['dist2'],
    'quad_cov12': res_quad.cov_params().loc['d_dist_mi', 'dist2'],
    'quad_p2': res_quad.pvalues['dist2'], 'f_quad_p': float(f_test_quad.pvalue),
}
import json
with open(os.path.join(DATA, 'doseresponse_coefs_new.json'), 'w') as f:
    json.dump(out, f, indent=2)
main.to_csv(os.path.join(DATA, 'doseresponse_panel_new.csv'), index=False)
print("\nSaved doseresponse_coefs_new.json, doseresponse_panel_new.csv")
