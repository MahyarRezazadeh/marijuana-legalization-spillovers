# ============================================================
# Diagnostic: does controlling for region x transition fixed
# effects absorb the pre-trend detected in analyze_pretrends.py?
# If eventual-switcher counties cluster in specific regions whose
# overdose trajectories were already diverging for reasons
# unrelated to marijuana policy (e.g. the geographic spread of
# fentanyl), region x transition FE should soak that up.
# ============================================================
import os
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

stacked = pd.read_csv(os.path.join(DATA, 'pretrends_panel_rec.csv'))

REGION = {
    'CT':'Northeast','ME':'Northeast','MA':'Northeast','NH':'Northeast','RI':'Northeast','VT':'Northeast','NJ':'Northeast','NY':'Northeast','PA':'Northeast',
    'IL':'Midwest','IN':'Midwest','MI':'Midwest','OH':'Midwest','WI':'Midwest','IA':'Midwest','KS':'Midwest','MN':'Midwest','MO':'Midwest','NE':'Midwest','ND':'Midwest','SD':'Midwest',
    'DE':'South','FL':'South','GA':'South','MD':'South','NC':'South','SC':'South','VA':'South','DC':'South','WV':'South','AL':'South','KY':'South','MS':'South','TN':'South','AR':'South','LA':'South','OK':'South','TX':'South',
    'AZ':'West','CO':'West','ID':'West','MT':'West','NV':'West','NM':'West','UT':'West','WY':'West','AK':'West','CA':'West','HI':'West','OR':'West','WA':'West',
}
stacked['region'] = stacked['state_abbr'].map(REGION)
print("Region coverage check (unmapped states):", stacked[stacked['region'].isna()]['state_abbr'].unique())
print(stacked.groupby(['bucket', 'region']).size().unstack(fill_value=0))

stacked['g'] = pd.Categorical(stacked['state_abbr']).codes

print("\n=== With region x transition fixed effects ===")
model2 = smf.ols('d_od ~ lead2plus + lead1 + switch0 + C(transition):C(region)', data=stacked).fit(
    cov_type='cluster', cov_kwds={'groups': stacked['g']})
for term in ['lead2plus', 'lead1', 'switch0']:
    print(f"{term:12s} b={model2.params[term]:+.4f} se={model2.bse[term]:.4f} p={model2.pvalues[term]:.4f}")
f2 = model2.f_test('lead1 = 0, lead2plus = 0')
print(f"Joint F-test (leads=0): F={float(f2.fvalue):.4f} p={float(f2.pvalue):.4f}")

print("\n=== With region fixed effects only (not interacted with transition) ===")
model3 = smf.ols('d_od ~ lead2plus + lead1 + switch0 + C(transition) + C(region)', data=stacked).fit(
    cov_type='cluster', cov_kwds={'groups': stacked['g']})
for term in ['lead2plus', 'lead1', 'switch0']:
    print(f"{term:12s} b={model3.params[term]:+.4f} se={model3.bse[term]:.4f} p={model3.pvalues[term]:.4f}")
f3 = model3.f_test('lead1 = 0, lead2plus = 0')
print(f"Joint F-test (leads=0): F={float(f3.fvalue):.4f} p={float(f3.pvalue):.4f}")

# Which regions dominate the switcher/lead pool vs. control pool?
print("\nShare of each bucket by region:")
print(pd.crosstab(stacked['bucket'], stacked['region'], normalize='index').round(3))
