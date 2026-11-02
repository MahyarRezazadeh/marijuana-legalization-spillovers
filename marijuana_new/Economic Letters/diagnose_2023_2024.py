# ============================================================
# Deep dive on the 2023-2024 sign reversal: is it a regional
# confound (Appalachian/Midwest counties on their own worsening
# trajectory, independent of marijuana access), an outlier-driven
# artifact, or a genuine nonlinearity in the mean-reversion
# relationship?
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)
s = s.dropna(subset=['baseline_rate'])
s['g'] = pd.Categorical(s['state_abbr']).codes

sub = s[s['transition'] == '2023-2024'].copy()
print(f"2023-2024 sample: {len(sub)} rows, {int(sub['switcher'].sum())} switchers")

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
sub['division'] = sub['state_abbr'].map(DIVISION)
print("\nSwitcher counties by state (2023-2024):")
print(sub[sub['switcher']==1]['state_abbr'].value_counts())
print("\nSwitcher counties by division vs all controls by division:")
print(pd.crosstab(sub['switcher'], sub['division']))

# 1. Region/division control
m_div = smf.ols('d_od ~ switcher + baseline_rate + C(division)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['g']})
print(f"\n[1] + division FE: switcher b={m_div.params['switcher']:+.4f} p={m_div.pvalues['switcher']:.4f}")

m_divstate = smf.ols('d_od ~ switcher + baseline_rate + C(state_abbr)', data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['g']})
print(f"[1b] + state FE: switcher b={m_divstate.params['switcher']:+.4f} p={m_divstate.pvalues['switcher']:.4f}")

# 2. Outlier / leverage diagnostics
ols_plain = sm.OLS(sub['d_od'], sm.add_constant(sub[['switcher', 'baseline_rate']])).fit()
infl = ols_plain.get_influence()
sub['cooks_d'] = infl.cooks_distance[0]
print("\n[2] Top 10 by Cook's distance:")
print(sub.nlargest(10, 'cooks_d')[['fips5', 'state_abbr', 'switcher', 'baseline_rate', 'd_od', 'cooks_d']].to_string(index=False))

trimmed = sub[sub['cooks_d'] < sub['cooks_d'].quantile(0.95)]
m_trim = smf.ols('d_od ~ switcher + baseline_rate', data=trimmed).fit(cov_type='cluster', cov_kwds={'groups': trimmed['g']})
print(f"\n[2b] Excluding top 5% Cook's D (n={len(trimmed)}): switcher b={m_trim.params['switcher']:+.4f} p={m_trim.pvalues['switcher']:.4f}")

# 3. Winsorize baseline_rate at 95th percentile
cap = sub['baseline_rate'].quantile(0.95)
sub['baseline_rate_wins'] = sub['baseline_rate'].clip(upper=cap)
m_wins = smf.ols('d_od ~ switcher + baseline_rate_wins', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
print(f"\n[3] Winsorized baseline_rate at p95: switcher b={m_wins.params['switcher']:+.4f} p={m_wins.pvalues['switcher']:.4f}")

# 4. Division FE + winsorized baseline
m_divwins = smf.ols('d_od ~ switcher + baseline_rate_wins + C(division)', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
print(f"[4] + division FE + winsorized baseline: switcher b={m_divwins.params['switcher']:+.4f} p={m_divwins.pvalues['switcher']:.4f}")

# 5. Raw group means, no regression -- descriptive sanity check
print("\n[5] Raw group means (2023-2024): d_od by switcher status")
print(sub.groupby('switcher')['d_od'].agg(['mean', 'median', 'std', 'count']))
