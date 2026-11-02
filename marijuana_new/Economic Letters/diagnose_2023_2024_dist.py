import os
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
s = s[s['d_od'].notna() & (s['population'] > 0)].copy()
s['baseline_rate'] = s['od_2020'] / (s['population'] / 1e5)
s = s.dropna(subset=['baseline_rate'])
s['g'] = pd.Categorical(s['state_abbr']).codes

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

m0 = smf.ols('d_od ~ d_dist_mi + baseline_rate', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
print(f"No region control: dist b={m0.params['d_dist_mi']:+.5f} p={m0.pvalues['d_dist_mi']:.4f}")

m1 = smf.ols('d_od ~ d_dist_mi + baseline_rate + C(division)', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
print(f"+ division FE: dist b={m1.params['d_dist_mi']:+.5f} p={m1.pvalues['d_dist_mi']:.4f}")

cap = sub['baseline_rate'].quantile(0.95)
sub['baseline_rate_wins'] = sub['baseline_rate'].clip(upper=cap)
m2 = smf.ols('d_od ~ d_dist_mi + baseline_rate_wins + C(division)', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
print(f"+ division FE + winsorized baseline: dist b={m2.params['d_dist_mi']:+.5f} p={m2.pvalues['d_dist_mi']:.4f}")
