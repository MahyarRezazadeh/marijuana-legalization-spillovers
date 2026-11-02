# ============================================================
# PART B ANALYSIS: Marijuana vs alcohol arrests, before/after
# each legalizing state's OWN recreational legalization date.
# Staggered-adoption TWFE (state FE + year FE), clustered by state.
# ============================================================
import os
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

df = pd.read_csv(os.path.join(DATA, 'state_trends_mj_alcohol.csv'))
df['year'] = df['month'].str[3:7].astype(int)
df['mon']  = df['month'].str[0:2].astype(int)

# Aggregate monthly -> annual (mean monthly rate per year; robust to a few missing months)
annual = (df.groupby(['state_abbr', 'rec_year', 'offense', 'year'])['rate_per_100k']
          .mean().reset_index())

wide = annual.pivot_table(index=['state_abbr', 'rec_year', 'year'],
                           columns='offense', values='rate_per_100k').reset_index()
wide['post'] = (wide['year'] >= wide['rec_year']).astype(int)

print(f"Panel: {wide['state_abbr'].nunique()} states x {wide['year'].nunique()} years = {len(wide)} obs")
print(wide.groupby('post')[['marijuana', 'alcohol']].mean())

wide = wide.set_index(['state_abbr', 'year'])

def run_twfe(outcome):
    mod = PanelOLS.from_formula(f"{outcome} ~ post + EntityEffects + TimeEffects", data=wide)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    return res

res_mj  = run_twfe('marijuana')
res_alc = run_twfe('alcohol')

print("\n=== Marijuana arrest rate (state+year FE, clustered by state) ===")
print(res_mj.summary)
print("\n=== Alcohol arrest rate (state+year FE, clustered by state) ===")
print(res_alc.summary)

# Save key numbers
summary = pd.DataFrame({
    'outcome': ['marijuana', 'alcohol'],
    'beta_post': [res_mj.params['post'], res_alc.params['post']],
    'se': [res_mj.std_errors['post'], res_alc.std_errors['post']],
    'pvalue': [res_mj.pvalues['post'], res_alc.pvalues['post']],
    'nobs': [res_mj.nobs, res_alc.nobs],
})
summary.to_csv(os.path.join(DATA, 'partB_twfe_results.csv'), index=False)
print("\n", summary)

wide.reset_index().to_csv(os.path.join(DATA, 'partB_annual_panel.csv'), index=False)

# ── Event-study: align states by years-since-own-legalization ──
ev = wide.reset_index().copy()
ev['event_year'] = ev['year'] - ev['rec_year']
event_agg = (ev[ev['event_year'].between(-9, 9)]
             .groupby('event_year')[['marijuana', 'alcohol']]
             .agg(['mean', 'count']))
event_agg.columns = ['_'.join(c) for c in event_agg.columns]
event_agg.to_csv(os.path.join(DATA, 'partB_event_study.csv'))
print("\nEvent-study (years since own legalization):")
print(event_agg)

