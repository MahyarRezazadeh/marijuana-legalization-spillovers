# ============================================================
# Binary (switcher) treatment under three fixed-effects specs,
# requested as a standalone report (not folded into main.tex):
#   1. Time (transition) FE only
#   2. State x Time FE (full state-by-transition cell dummies,
#      C(transition):C(state_abbr) -- nests plain time FE, since
#      the state-year cells span the same subspace)
#   3. County FE, separately, alongside time FE
#
# Panel: stacked_panel_rec_full.csv, the same panel behind Table 1/2
# throughout this project. Treatment ("switcher") is defined off the
# ACTUAL retail-sales/dispensary-opening date (legal_dates_new.csv's
# retail_date column, not the earlier policy-effective date), per
# the project's standing 2026-07-22 correction -- confirmed here,
# not re-derived.
#
# No baseline-rate (mean-reversion) control here, matching what was
# actually requested this round -- these are the raw FE comparisons,
# not the paper's "preferred" mean-reversion-adjusted spec.
#
# Clustered by state (g), matching every other FE-comparison table
# already built in this project (reestimate_main_with_controls.py,
# analyze_countyFE.py) -- flagged explicitly since the previous
# event-study check used county clustering instead, per that
# request specifically.
# ============================================================
import os
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['g'] = pd.Categorical(s['state_abbr']).codes

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'))
print(f"Treatment confirmed as actual retail/dispensary-opening date (legal_dates_new.csv 'retail_date' column,\n"
      f"not policy-effective date) -- {legal['retail_date'].notna().sum()} states with a retail date on file.\n")

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
