# ============================================================
# Full coefficient report for the border-adjacency panel's
# "+ state x transition FE" spec (the one row in
# border_robustness_results.csv that flipped sign and reached
# significance under state-clustered SEs) -- not just the switcher
# coefficient, but every other estimated coefficient too. State x
# transition fixed effects themselves (~150 individual cell dummies)
# are estimated but not individually listed, matching standard
# practice and this project's own precedent (analyze_countyFE_full.py).
# ============================================================
import os
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_border.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s = s.dropna(subset=['baseline_rate'])
s['g'] = pd.Categorical(s['state_abbr']).codes

TRANSITIONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
s['transition'] = pd.Categorical(s['transition'], categories=TRANSITIONS, ordered=True)

m = smf.ols('d_od ~ switcher + baseline_rate + C(transition):C(state_abbr)', data=s).fit(
    cov_type='cluster', cov_kwds={'groups': s['g']})

fe_terms = [t for t in m.params.index if t not in ('Intercept', 'switcher', 'baseline_rate')]
n_main_state = sum(1 for t in fe_terms if t.startswith('C(state_abbr)[') and ':' not in t)
n_interaction = sum(1 for t in fe_terms if ':' in t)
print(f"N = {int(m.nobs)}, states = {s['state_abbr'].nunique()}, transitions = {s['transition'].nunique()}")
print(f"Patsy coded this as {n_main_state} state main-effect dummies + {n_interaction} transition:state "
      f"interaction dummies ({len(fe_terms)} total FE terms, {m.df_model+1:.0f} model df) -- "
      f"together these ARE the full state x transition cell fixed effects, just not full-rank-per-cell coded.")
print(f"R-squared (mostly mechanical, from {len(fe_terms)} FE dummies): {m.rsquared:.4f}\n")

rows = []
for term in ['Intercept', 'switcher', 'baseline_rate']:
    if term not in m.params.index:
        continue
    label = {'switcher': 'Switcher (border-adjacent, coefficient of interest)',
              'baseline_rate': 'Baseline (2020) overdose rate'}.get(term, term)
    rows.append(dict(term=label, coef=m.params[term], se=m.bse[term], p=m.pvalues[term]))
    ci_lo = m.params[term] - 1.96 * m.bse[term]
    ci_hi = m.params[term] + 1.96 * m.bse[term]
    print(f"{label:50s} b={m.params[term]:+.5f}  se={m.bse[term]:.5f}  p={m.pvalues[term]:.4f}  "
          f"95% CI=[{ci_lo:+.5f}, {ci_hi:+.5f}]")

n_sig_fe = sum(1 for t in fe_terms if m.pvalues[t] < 0.05)
print(f"[{len(fe_terms)} state x transition fixed-effect terms estimated, not individually listed "
      f"-- {n_sig_fe} individually significant at p<0.05]")

pd.DataFrame(rows).to_csv(os.path.join(DATA, 'border_statextime_full_coefficients.csv'), index=False)
print("\nSaved border_statextime_full_coefficients.csv")
