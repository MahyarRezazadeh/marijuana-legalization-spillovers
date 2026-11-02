# ============================================================
# Follow-up to analyze_eventstudy_full.py: the per-transition
# (year-over-year) coefficients in that check are each a SEPARATE
# one-year marginal effect, not a cumulative effect since switching.
# A positive lag coefficient does not by itself mean mortality ends
# up higher than before switching -- it means the decline partially
# reverses THAT year. This script computes the CUMULATIVE effect
# (switch0, switch0+p1, switch0+p1+p2, switch0+p1+p2+p3) using the
# fitted model's own covariance matrix (via t_test linear
# combinations), not a naive sum of point estimates/SEs, so the
# standard errors correctly account for covariance between the
# event-time dummies.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'eventstudy_full_panel_rec.csv'), dtype={'fips5': str})

EVENT_BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
rhs_terms = ' + '.join(EVENT_BINS)

for spec_name, formula, df in [
    ('raw', f'd_od ~ {rhs_terms} + C(transition)', s),
    ('adjusted (preferred)', f'd_od ~ {rhs_terms} + baseline_rate + C(transition)', s.dropna(subset=['baseline_rate'])),
]:
    m = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['county_g']})
    print(f"\n=== {spec_name} ===")
    cum_terms = ['et_0', 'et_0 + et_p1', 'et_0 + et_p1 + et_p2', 'et_0 + et_p1 + et_p2 + et_p3']
    cum_labels = ['at switch (t=0)', 'cumulative through t=+1', 'cumulative through t=+2', 'cumulative through t=+3']
    for term, label in zip(cum_terms, cum_labels):
        tt = m.t_test(term)
        coef = float(np.asarray(tt.effect).ravel()[0])
        se = float(np.asarray(tt.sd).ravel()[0])
        p = float(np.asarray(tt.pvalue).ravel()[0])
        print(f"  {label:28s}: b={coef:+.4f}  se={se:.4f}  p={p:.4f}  95% CI=[{coef-1.96*se:+.4f}, {coef+1.96*se:+.4f}]")
