import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)
s = s.dropna(subset=['baseline_rate'])
s['g'] = pd.Categorical(s['state_abbr']).codes

VIABLE = ['2021-2022', '2022-2023', '2023-2024']
for i, t in enumerate(VIABLE, start=1):
    s[f'sw_t{i}'] = ((s['transition'] == t) * s['switcher']).astype(float)
    s[f'dist_t{i}'] = ((s['transition'] == t) * s['d_dist_mi']).astype(float)
    s[f'base_t{i}'] = ((s['transition'] == t) * s['baseline_rate']).astype(float)

# Let mean-reversion strength ALSO vary by period (fully flexible model)
rhs = ['C(transition)'] + [f'sw_t{i}' for i in [1,2,3]] + [f'dist_t{i}' for i in [1,2,3]] + [f'base_t{i}' for i in [1,2,3]]
m = smf.ols('d_od ~ ' + ' + '.join(rhs), data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g']})
print("=== Period-varying switcher, distance, AND baseline-rate coefficients ===")
for i, t in enumerate(VIABLE, start=1):
    print(f"{t}: switcher b={m.params[f'sw_t{i}']:+.4f} p={m.pvalues[f'sw_t{i}']:.4f} | "
          f"dist b={m.params[f'dist_t{i}']:+.5f} p={m.pvalues[f'dist_t{i}']:.4f} | "
          f"baseline_rate_coef={m.params[f'base_t{i}']:+.4f} p={m.pvalues[f'base_t{i}']:.4f}")

# Also: quadratic/nonlinear baseline_rate control within 2023-2024 -- is the
# relationship between baseline rate and Δod actually linear, or does a
# lienar control over-correct for very high-baseline counties?
sub2324 = s[s['transition'] == '2023-2024'].copy()
sub2324['baseline_rate2'] = sub2324['baseline_rate'] ** 2
m_quad = smf.ols('d_od ~ switcher + baseline_rate + baseline_rate2', data=sub2324).fit(
    cov_type='cluster', cov_kwds={'groups': sub2324['g']})
print(f"\n2023-2024 with quadratic baseline_rate control: switcher b={m_quad.params['switcher']:+.4f} "
      f"p={m_quad.pvalues['switcher']:.4f}")
print(f"baseline_rate linear term: b={m_quad.params['baseline_rate']:+.4f} p={m_quad.pvalues['baseline_rate']:.4f}")
print(f"baseline_rate quadratic term: b={m_quad.params['baseline_rate2']:+.6f} p={m_quad.pvalues['baseline_rate2']:.4f}")

# Distribution check: how extreme are the high-baseline switchers?
print(f"\n2023-2024 baseline_rate distribution by switcher status:")
print(sub2324.groupby('switcher')['baseline_rate'].describe())
