# ============================================================
# Rebuilt in New version EL/ -- same three-spec binary-switcher FE
# comparison as before, now using this folder's own parametrized
# main panel (data/stacked_panel_main_t50.csv, T=50, matching the
# project's standard threshold) instead of the parent folder's
# stacked_panel_rec_full.csv. Treatment ("switcher") is defined off
# the actual retail-sales/dispensary-opening date (legal_dates_new.csv
# 'retail_date'), confirmed, not re-derived.
#
# No baseline-rate control (raw FE comparison, as requested).
# Clustered by state, matching every other FE-comparison table in
# this project (analyze_main_param.py, analyze_dispensary_robustness.py).
# ============================================================
import os
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
PARENT_DATA = os.path.join(BASE, '..', 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_main_t50.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['g'] = pd.Categorical(s['state_abbr']).codes

legal = pd.read_csv(os.path.join(PARENT_DATA, 'legal_dates_new.csv'))
print(f"Treatment confirmed as actual retail/dispensary-opening date (legal_dates_new.csv 'retail_date' column) --\n"
      f"{legal['retail_date'].notna().sum()} states with a retail date on file.\n")

specs = [
    ('1. Time (transition) FE only', 'd_od ~ switcher + C(transition)'),
    ('2. State x Time FE (nests time FE)', 'd_od ~ switcher + C(transition):C(state_abbr)'),
    ('3. County FE + Time FE (separately)', 'd_od ~ switcher + C(transition) + C(fips5)'),
]

rows = []
for name, formula in specs:
    m = smf.ols(formula, data=s).fit(cov_type='cluster', cov_kwds={'groups': s['g']})
    rows.append(dict(spec=name, coef=m.params['switcher'], se=m.bse['switcher'],
                      p=m.pvalues['switcher'], n=int(m.nobs), r2=m.rsquared))
    ci_lo = m.params['switcher'] - 1.96 * m.bse['switcher']
    ci_hi = m.params['switcher'] + 1.96 * m.bse['switcher']
    print(f"{name}")
    print(f"    switcher: b={m.params['switcher']:+.4f}  se={m.bse['switcher']:.4f}  "
          f"p={m.pvalues['switcher']:.4f}  95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
          f"n={int(m.nobs)}  R2={m.rsquared:.4f}\n")

pd.DataFrame(rows).to_csv(os.path.join(DATA, 'binary_fe_report.csv'), index=False)
print("Saved binary_fe_report.csv")
