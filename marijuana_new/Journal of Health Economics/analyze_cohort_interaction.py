# ============================================================
# Formal, single-regression version of the cohort-composition check
# (analyze_cohort_composition.py ran 3 separate regressions; this
# pools them into one, with full event-time x cohort interaction
# terms replacing the plain pooled event-time dummies, sharing one
# baseline_rate coefficient and the full control group across
# cohorts). A plain ADDITIVE cohort fixed effect is not separately
# identifiable here -- EventTime = Transition - Cohort is an exact
# arithmetic identity (the age-period-cohort problem), so a cohort
# main effect would be perfectly collinear with the existing
# transition and event-time dummies. Interaction terms are not
# subject to that collinearity and are what actually let cohort-
# specific responses show up.
#
# Adds a formal joint F-test for poolability: (a) targeted test on
# t=+1 specifically (the coefficient the paper's cohort claim is
# about), (b) broader test across all three event times common to
# every cohort (-1, 0, +1).
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

panel = pd.read_csv(os.path.join(DATA, 'eventstudy_main_panel.csv'), dtype={'fips5': str})

et0 = panel[panel['bucket'] == 'et_0'][['fips5', 'transition']].copy()
et0['switch_year'] = et0['transition'].str.split('-').str[1].astype(int)
switch_year_map = et0.set_index('fips5')['switch_year'].to_dict()
panel['switch_year'] = panel['fips5'].map(switch_year_map)

EVENT_BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
COHORTS = [2022, 2023, 2024]

sub = panel.dropna(subset=['baseline_rate']).copy()
sub['county_g'] = pd.Categorical(sub['fips5']).codes

# interaction dummies: et_{k}_{g} = 1 if county is in cohort g AND bucket k;
# only create columns for (k,g) combinations that actually occur (matching
# each cohort's observable event-time window)
interaction_cols = []
for g in COHORTS:
    cohort_fips = set(et0.loc[et0['switch_year'] == g, 'fips5'])
    for b in EVENT_BINS:
        col = f'{b}_c{g}'
        mask = (sub['bucket'] == b) & (sub['fips5'].isin(cohort_fips))
        if mask.sum() == 0:
            continue
        sub[col] = mask.astype(float)
        interaction_cols.append(col)

print(f"Built {len(interaction_cols)} event-time x cohort interaction terms: {interaction_cols}")

rhs = ' + '.join(interaction_cols)
m = smf.ols(f'd_od ~ {rhs} + baseline_rate + C(transition)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['county_g']})

print(f"\nN = {int(m.nobs)}")
print("\n=== Pooled regression with full event-time x cohort interactions ===")
for col in interaction_cols:
    print(f"  {col}: b={m.params[col]:+.4f} se={m.bse[col]:.4f} p={m.pvalues[col]:.4f}")

# sanity check: these should match Table 6's separate-regression coefficients closely
# (small differences possible since baseline_rate and transition FE are now shared/pooled)

# --- targeted F-test: is the t=+1 coefficient equal across all 3 cohorts? ---
p1_terms = [f'et_p1_c{g}' for g in COHORTS if f'et_p1_c{g}' in m.params.index]
print(f"\nt=+1 terms present: {p1_terms}")
restriction_p1 = ', '.join([f'{p1_terms[i]} = {p1_terms[i+1]}' for i in range(len(p1_terms) - 1)])
f_p1 = m.f_test(restriction_p1)
print(f"\n=== Joint F-test: t=+1 coefficient equal across cohorts ===")
print(f"  H0: {restriction_p1}")
print(f"  F={float(f_p1.fvalue):.4f}  p={float(f_p1.pvalue):.6f}")

# --- broader test: all three event times common to every cohort (-1, 0, +1) ---
shared_bins = ['et_m1', 'et_0', 'et_p1']
restrictions = []
for b in shared_bins:
    terms = [f'{b}_c{g}' for g in COHORTS if f'{b}_c{g}' in m.params.index]
    for i in range(len(terms) - 1):
        restrictions.append(f'{terms[i]} = {terms[i+1]}')
restriction_all = ', '.join(restrictions)
f_all = m.f_test(restriction_all)
print(f"\n=== Joint F-test: ALL shared event times (t=-1,0,+1) equal across cohorts ===")
print(f"  H0: {restriction_all}")
print(f"  F={float(f_all.fvalue):.4f}  p={float(f_all.pvalue):.6f}")

out = dict(
    n=int(m.nobs),
    coefficients={col: dict(coef=float(m.params[col]), se=float(m.bse[col]), p=float(m.pvalues[col]))
                  for col in interaction_cols},
    f_test_t1_equal=dict(F=float(f_p1.fvalue), p=float(f_p1.pvalue), restriction=restriction_p1),
    f_test_all_shared_equal=dict(F=float(f_all.fvalue), p=float(f_all.pvalue), restriction=restriction_all),
)
with open(os.path.join(DATA, 'cohort_interaction_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved cohort_interaction_results.json")
