# ============================================================
# Dose-response / marginal-effect analysis for the overdose
# continuous-distance specification. Tests whether a quadratic
# (and cubic, as a further check) term in distance-change is
# statistically warranted before treating the marginal effect
# as anything other than the constant linear slope.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel.csv'))
main = panel[panel['sample_grp'] != 'Always near rec border'].copy()
main = main[main['d_od'].notna()].copy()
print(f"N = {len(main)}")
print(main['dist_change_mi'].describe())

def cluster_fit(formula, df):
    mod = smf.ols(formula, data=df)
    g = pd.Categorical(df['state_abbr']).codes
    return mod.fit(cov_type='cluster', cov_kwds={'groups': g})

# ── Linear (baseline, already reported) ──
res_lin = cluster_fit('d_od ~ dist_change_mi', main)

# ── Quadratic ──
main['dist2'] = main['dist_change_mi'] ** 2
res_quad = cluster_fit('d_od ~ dist_change_mi + dist2', main)

# ── Cubic ──
main['dist3'] = main['dist_change_mi'] ** 3
res_cubic = cluster_fit('d_od ~ dist_change_mi + dist2 + dist3', main)

print("\n=== LINEAR ===")
print(res_lin.summary().tables[1])
print(f"AIC={res_lin.aic:.1f}  BIC={res_lin.bic:.1f}")

print("\n=== QUADRATIC ===")
print(res_quad.summary().tables[1])
print(f"AIC={res_quad.aic:.1f}  BIC={res_quad.bic:.1f}")
print(f"Quadratic term p-value: {res_quad.pvalues['dist2']:.4f}")

print("\n=== CUBIC ===")
print(res_cubic.summary().tables[1])
print(f"AIC={res_cubic.aic:.1f}  BIC={res_cubic.bic:.1f}")

# Joint F-test: are the nonlinear terms jointly significant vs. linear?
f_test_quad = res_quad.f_test('dist2 = 0')
f_test_cubic = res_cubic.f_test('dist2 = 0, dist3 = 0')
print(f"\nF-test (quadratic term = 0): F={float(f_test_quad.fvalue):.3f}, p={float(f_test_quad.pvalue):.4f}")
print(f"F-test (quadratic+cubic terms = 0): F={float(f_test_cubic.fvalue):.3f}, p={float(f_test_cubic.pvalue):.4f}")

# Save coefficients for plotting
out = {
    'lin_b1': res_lin.params['dist_change_mi'], 'lin_se1': res_lin.bse['dist_change_mi'],
    'quad_b1': res_quad.params['dist_change_mi'], 'quad_se1': res_quad.bse['dist_change_mi'],
    'quad_b2': res_quad.params['dist2'], 'quad_se2': res_quad.bse['dist2'],
    'quad_cov12': res_quad.cov_params().loc['dist_change_mi', 'dist2'],
    'quad_p2': res_quad.pvalues['dist2'],
}
pd.Series(out).to_json(os.path.join(DATA, 'doseresponse_coefs.json'))
main.to_csv(os.path.join(DATA, 'doseresponse_panel.csv'), index=False)
print("\nSaved doseresponse_coefs.json, doseresponse_panel.csv")
