# ============================================================
# Event study (by state market age) for revenue PER STORE, border vs
# interior, 70mi threshold -- checks whether the significant premium
# found in the pooled panel (analyze_revenue_perstore_border70.py) is
# stable across a state's market lifecycle or concentrated in specific
# years, mirroring analyze_land_value_border.py's market-age event
# study and the original analyze_revenue_border_eventstudy.py.
# event_year = fiscal/calendar year - state's own retail-launch year
# (CO launched 2014-01-01 -> event_year = fy_year - 2014; WA launched
# 2014-07-08, so its first full fiscal year of operation is FY2015 ->
# event_year = fy_year - 2015, same convention used throughout this
# project for aligning CO's calendar-year and WA's fiscal-year data).
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

panel = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel.csv'))
launch_offset = {'CO': 2014, 'WA': 2015}
panel['event_year'] = panel.apply(lambda r: int(r['fy_year']) - launch_offset[r['state_abbr']], axis=1)

print("Event-year coverage by state:")
print(panel.groupby(['state_abbr', 'event_year']).size().unstack(fill_value=0))

# bin sparse tail years together, same convention as the original revenue event study
panel['event_year_bin'] = panel['event_year'].clip(upper=9)
panel.loc[panel['event_year_bin'] >= 9, 'event_year_bin'] = 9

panel['ey'] = pd.Categorical(panel['event_year_bin'])
panel['g'] = pd.Categorical(panel['fips5']).codes

formula = 'log_revenue_per_store ~ log_population + C(state_abbr) + C(ey) + border_70mi:C(ey)'
m = smf.ols(formula, data=panel).fit(cov_type='HC1')
mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['g']})

print("\n" + "=" * 70)
print("EVENT STUDY BY MARKET AGE -- border_70mi x event-year interaction")
print("outcome: log(revenue per store)")
print("=" * 70)
rows = []
for ey in sorted(panel['event_year_bin'].unique()):
    term = f'border_70mi:C(ey)[{ey}]'
    if term not in m.params.index:
        continue
    b, se, p = m.params[term], m.bse[term], m.pvalues[term]
    se_c, p_c = mc.bse[term], mc.pvalues[term]
    label = f'{ey}+' if ey == 9 else str(ey)
    n_this = int((panel['event_year_bin'] == ey).sum())
    n_border_this = int(((panel['event_year_bin'] == ey) & (panel['border_70mi'] == 1)).sum())
    print(f"market-age year {label:>3s} (n={n_this:4d}, border obs={n_border_this:3d})   "
          f"b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f}")
    rows.append(dict(event_year=label, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c,
                      n=n_this, n_border=n_border_this))

print(f"\nR2={m.rsquared:.3f} n={int(m.nobs)}")
pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'revenue_perstore_eventstudy_results.csv'), index=False)
print("\nSaved revenue_perstore_eventstudy_results.csv")
