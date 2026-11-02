# ============================================================
# County-clustered companion to the parent folder's
# diagnose_2023_2024.py / diagnose_2023_2024_dist.py: reproduces
# the Table robustsummary Panel B switcher/distance rows (division
# FE, winsorized baseline, Cook's-distance trim) on the local
# 50-mile panel, adding a county-clustered SE/p alongside the
# original state-clustered one. Point estimates match the parent
# script exactly (same data); only the clustering level differs.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_main_t50.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s = s.dropna(subset=['baseline_rate'])
s['g'] = pd.Categorical(s['state_abbr']).codes
s['g_county'] = pd.Categorical(s['fips5']).codes

DIVISION = {
    'CT':'NewEngland','ME':'NewEngland','MA':'NewEngland','NH':'NewEngland','RI':'NewEngland','VT':'NewEngland',
    'NJ':'MidAtlantic','NY':'MidAtlantic','PA':'MidAtlantic',
    'IL':'ENCentral','IN':'ENCentral','MI':'ENCentral','OH':'ENCentral','WI':'ENCentral',
    'IA':'WNCentral','KS':'WNCentral','MN':'WNCentral','MO':'WNCentral','NE':'WNCentral','ND':'WNCentral','SD':'WNCentral',
    'DE':'SAtlantic','FL':'SAtlantic','GA':'SAtlantic','MD':'SAtlantic','NC':'SAtlantic','SC':'SAtlantic','VA':'SAtlantic','DC':'SAtlantic','WV':'SAtlantic',
    'AL':'ESCentral','KY':'ESCentral','MS':'ESCentral','TN':'ESCentral',
    'AR':'WSCentral','LA':'WSCentral','OK':'WSCentral','TX':'WSCentral',
    'AZ':'Mountain','CO':'Mountain','ID':'Mountain','MT':'Mountain','NV':'Mountain','NM':'Mountain','UT':'Mountain','WY':'Mountain',
    'AK':'Pacific','CA':'Pacific','HI':'Pacific','OR':'Pacific','WA':'Pacific',
}
sub = s[s['transition'] == '2023-2024'].copy()
sub['division'] = sub['state_abbr'].map(DIVISION)
print(f"2023-2024 sample: {len(sub)} rows, {int(sub['switcher'].sum())} switchers, "
      f"{sub['g_county'].nunique()} county clusters")

def fit_both(formula, data):
    m = smf.ols(formula, data=data).fit(cov_type='cluster', cov_kwds={'groups': data['g']})
    m_c = smf.ols(formula, data=data).fit(cov_type='cluster', cov_kwds={'groups': data['g_county']})
    return m, m_c

rows = []

# 0. "Baseline-rate control only" reproduces the period-interaction
# model's own 2023-2024 coefficient (Table period / period_interactions.csv),
# NOT a simple regression re-run on just the 691-row subsample -- those
# differ because the period-interaction model's baseline_rate coefficient
# is identified off all five transitions, not just this one. Pull it
# from period_interactions.csv (must be regenerated first by
# analyze_period_interactions.py) rather than recomputing, to stay
# numerically consistent with Table period.
pi = pd.read_csv(os.path.join(BASE, 'tables', 'period_interactions.csv'))
pi3 = pi[pi['transition'] == '2023-2024'].iloc[0]
rows.append(dict(spec="Baseline-rate control only",
                  sw_coef=pi3['sw_coef'], sw_p=pi3['sw_p'], sw_p_county=pi3['sw_p_county'], sw_se_county=pi3['sw_se_county'],
                  dist_coef=pi3['dist_coef'], dist_p=pi3['dist_p'], dist_p_county=pi3['dist_p_county'], dist_se_county=pi3['dist_se_county']))

# 1. + division FE -- switcher AND distance
m1s, m1sc = fit_both('d_od ~ switcher + baseline_rate + C(division)', sub)
m1d, m1dc = fit_both('d_od ~ d_dist_mi + baseline_rate + C(division)', sub)
rows.append(dict(spec="+ Census-division fixed effects",
                  sw_coef=m1s.params['switcher'], sw_p=m1s.pvalues['switcher'], sw_p_county=m1sc.pvalues['switcher'], sw_se_county=m1sc.bse['switcher'],
                  dist_coef=m1d.params['d_dist_mi'], dist_p=m1d.pvalues['d_dist_mi'], dist_p_county=m1dc.pvalues['d_dist_mi'], dist_se_county=m1dc.bse['d_dist_mi']))

# 2. Winsorized baseline_rate at p95 -- switcher only (per Table robustsummary, distance = "---")
cap = sub['baseline_rate'].quantile(0.95)
sub['baseline_rate_wins'] = sub['baseline_rate'].clip(upper=cap)
m2s, m2sc = fit_both('d_od ~ switcher + baseline_rate_wins', sub)
rows.append(dict(spec="+ Winsorized baseline rate (95th pctile)",
                  sw_coef=m2s.params['switcher'], sw_p=m2s.pvalues['switcher'], sw_p_county=m2sc.pvalues['switcher'], sw_se_county=m2sc.bse['switcher'],
                  dist_coef=np.nan, dist_p=np.nan, dist_p_county=np.nan, dist_se_county=np.nan))

# 3. Division FE + winsorized baseline -- switcher AND distance
m3s, m3sc = fit_both('d_od ~ switcher + baseline_rate_wins + C(division)', sub)
m3d, m3dc = fit_both('d_od ~ d_dist_mi + baseline_rate_wins + C(division)', sub)
rows.append(dict(spec="+ Division FE & winsorized baseline",
                  sw_coef=m3s.params['switcher'], sw_p=m3s.pvalues['switcher'], sw_p_county=m3sc.pvalues['switcher'], sw_se_county=m3sc.bse['switcher'],
                  dist_coef=m3d.params['d_dist_mi'], dist_p=m3d.pvalues['d_dist_mi'], dist_p_county=m3dc.pvalues['d_dist_mi'], dist_se_county=m3dc.bse['d_dist_mi']))

# 4. Excluding top 5% by Cook's distance -- switcher only
ols_plain = sm.OLS(sub['d_od'], sm.add_constant(sub[['switcher', 'baseline_rate']])).fit()
infl = ols_plain.get_influence()
sub['cooks_d'] = infl.cooks_distance[0]
trimmed = sub[sub['cooks_d'] < sub['cooks_d'].quantile(0.95)]
m4s, m4sc = fit_both('d_od ~ switcher + baseline_rate', trimmed)
rows.append(dict(spec="Excluding top 5% by Cook's distance",
                  sw_coef=m4s.params['switcher'], sw_p=m4s.pvalues['switcher'], sw_p_county=m4sc.pvalues['switcher'], sw_se_county=m4sc.bse['switcher'],
                  dist_coef=np.nan, dist_p=np.nan, dist_p_county=np.nan, dist_se_county=np.nan))

for r in rows:
    print(f"{r['spec']:40s} sw b={r['sw_coef']:+.4f} p={r['sw_p']:.4f} p_county={r['sw_p_county']:.4f} | "
          f"dist b={'---' if pd.isna(r['dist_coef']) else format(r['dist_coef'], '+.5f')} "
          f"p={'---' if pd.isna(r['dist_p']) else format(r['dist_p'], '.4f')} "
          f"p_county={'---' if pd.isna(r['dist_p_county']) else format(r['dist_p_county'], '.4f')}")

pd.DataFrame(rows).to_csv(os.path.join(BASE, 'tables', 'robustsummary_panelB_countyclust.csv'), index=False)
print("\nSaved tables/robustsummary_panelB_countyclust.csv")
