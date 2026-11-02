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

rhs_no_baseline = ['C(transition)'] + [f'sw_t{i}' for i in [1,2,3]] + [f'dist_t{i}' for i in [1,2,3]]
m0 = smf.ols('d_od ~ ' + ' + '.join(rhs_no_baseline), data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g']})
print("=== WITHOUT baseline_rate control ===")
for i, t in enumerate(VIABLE, start=1):
    print(f"{t}: switcher b={m0.params[f'sw_t{i}']:+.4f} p={m0.pvalues[f'sw_t{i}']:.4f} | "
          f"dist b={m0.params[f'dist_t{i}']:+.5f} p={m0.pvalues[f'dist_t{i}']:.4f}")

# cross-check: plain single-period regression (like the original event_study_rec.csv), 2023-2024 only
sub2324 = s[s['transition'] == '2023-2024']
m_single = smf.ols('d_od ~ switcher', data=sub2324).fit(cov_type='cluster', cov_kwds={'groups': sub2324['g']})
print(f"\nCross-check, 2023-2024 alone (no baseline_rate, no other periods): "
      f"switcher b={m_single.params['switcher']:+.4f} p={m_single.pvalues['switcher']:.4f} n={int(m_single.nobs)}")

m_single_base = smf.ols('d_od ~ switcher + baseline_rate', data=sub2324).fit(cov_type='cluster', cov_kwds={'groups': sub2324['g']})
print(f"Cross-check, 2023-2024 alone (+ baseline_rate): "
      f"switcher b={m_single_base.params['switcher']:+.4f} p={m_single_base.pvalues['switcher']:.4f} n={int(m_single_base.nobs)}")

# check correlation between baseline_rate and switcher within 2023-2024
print(f"\nWithin 2023-2024: mean baseline_rate for switchers vs controls:")
print(sub2324.groupby('switcher')['baseline_rate'].agg(['mean', 'count']))

# check VIF-style: correlation between sw_t3 and baseline_rate in the full joint sample
print(f"\nCorrelation(sw_t3, baseline_rate) in joint sample: {s['sw_t3'].corr(s['baseline_rate']):.4f}")
print(f"Correlation(sw_t3, dist_t3) in joint sample: {s['sw_t3'].corr(s['dist_t3']):.4f}")
