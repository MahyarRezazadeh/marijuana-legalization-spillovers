# ============================================================
# Nonlinearity check on the new stacked multi-period recreational
# panel: adds quadratic/cubic terms in d_dist_mi to the pooled
# transition-FE regression and F-tests their joint significance.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['d_dist_mi2'] = s['d_dist_mi'] ** 2
s['d_dist_mi3'] = s['d_dist_mi'] ** 3
s['g'] = pd.Categorical(s['state_abbr']).codes

lin = smf.ols('d_od ~ d_dist_mi + C(transition)', data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g']})
quad = smf.ols('d_od ~ d_dist_mi + d_dist_mi2 + C(transition)', data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g']})
cubic = smf.ols('d_od ~ d_dist_mi + d_dist_mi2 + d_dist_mi3 + C(transition)', data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g']})

print("Linear:", lin.params['d_dist_mi'], lin.bse['d_dist_mi'], lin.pvalues['d_dist_mi'])
print("Quadratic term:", quad.params['d_dist_mi2'], quad.pvalues['d_dist_mi2'])

f_quad = quad.f_test('d_dist_mi2 = 0')
f_quadcubic = cubic.f_test('d_dist_mi2 = 0, d_dist_mi3 = 0')
print(f"F-test (quadratic only): F={float(f_quad.fvalue):.4f} p={float(f_quad.pvalue):.4f}")
print(f"F-test (quad+cubic): F={float(f_quadcubic.fvalue):.4f} p={float(f_quadcubic.pvalue):.4f}")

out = dict(
    linear=dict(coef=float(lin.params['d_dist_mi']), se=float(lin.bse['d_dist_mi']), p=float(lin.pvalues['d_dist_mi'])),
    quad_term=dict(coef=float(quad.params['d_dist_mi2']), se=float(quad.bse['d_dist_mi2']), p=float(quad.pvalues['d_dist_mi2'])),
    f_quad=dict(F=float(f_quad.fvalue), p=float(f_quad.pvalue)),
    f_quadcubic=dict(F=float(f_quadcubic.fvalue), p=float(f_quadcubic.pvalue)),
)
with open(os.path.join(DATA, 'doseresponse_multiperiod.json'), 'w') as f:
    json.dump(out, f, indent=2)

# save prediction grid for plotting
grid = pd.DataFrame({'d_dist_mi': np.linspace(s['d_dist_mi'].quantile(0.02), s['d_dist_mi'].quantile(0.98), 100)})
grid['d_dist_mi2'] = grid['d_dist_mi'] ** 2
grid['transition'] = s['transition'].mode()[0]
pred_lin = lin.get_prediction(grid[['d_dist_mi', 'transition']]).summary_frame()
pred_quad = quad.get_prediction(grid[['d_dist_mi', 'd_dist_mi2', 'transition']]).summary_frame()
grid['pred_lin'] = pred_lin['mean']; grid['lin_lo'] = pred_lin['mean_ci_lower']; grid['lin_hi'] = pred_lin['mean_ci_upper']
grid['pred_quad'] = pred_quad['mean']; grid['quad_lo'] = pred_quad['mean_ci_lower']; grid['quad_hi'] = pred_quad['mean_ci_upper']
grid.to_csv(os.path.join(DATA, 'doseresponse_grid_multiperiod.csv'), index=False)
print("Saved doseresponse_multiperiod.json, doseresponse_grid_multiperiod.csv")
