# ============================================================
# Event study: does the (null, on average) relationship between
# border-distance and cannabis revenue per capita change as each
# state's own market matures? Distance-to-border is time-invariant
# per county, so there's no "switcher" the way the main mortality
# design has one -- the natural event-time clock here is years
# since the STATE's own retail market opened (CO: 2014-01-01,
# WA: 2014-07-08), letting the two states line up on a common
# relative clock the same way the main paper aligns different
# real-calendar switch years onto event time.
#
# Estimated as: log_revenue_per_capita ~ C(event_year) +
#   log_dist_nonlegal x C(event_year) + log_population + C(state_abbr)
# i.e. one border-distance coefficient PER market-age year, not a
# single pooled coefficient. Primary spec uses continuous distance
# throughout -- NOT the binary <=50mi version, since that measure's
# 'border' side for CO is entirely one outlier county (Sedgwick,
# see analyze_revenue_border.py) and interacting it with event-time
# would just spread that same single-county artifact across bins.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

df = pd.read_csv(os.path.join(DATA, 'county_revenue_panel.csv'))
df = df[df['revenue_per_capita'] > 0].copy()
df['log_population'] = np.log(df['population'])

# event_year = years since the county's OWN state's retail market opened.
# CO's panel is calendar-year (launched 2014-01-01) -> event_year = fy_year - 2014.
# WA's panel is fiscal-year Jul-Jun (launched 2014-07-08, so FY2015 = first full FY
# of retail operation) -> event_year = fy_year - 2015.
launch_offset = {'CO': 2014, 'WA': 2015}
df['event_year'] = df.apply(lambda r: int(r['fy_year']) - launch_offset[r['state_abbr']], axis=1)

print("=" * 70)
print("Event-year coverage by state (years since own state's retail launch)")
print("=" * 70)
print(df.groupby(['state_abbr', 'event_year']).size().unstack(fill_value=0))

# bin sparse tail years (few counties observed that far out yet) together
df['event_year_bin'] = df['event_year'].clip(upper=10)
df.loc[df['event_year_bin'] >= 9, 'event_year_bin'] = 9  # 9 = "9+"

print("\n" + "=" * 70)
print("EVENT STUDY -- border-distance x event-year interaction")
print("outcome: log(revenue per capita); ref category: event_year_bin == 0")
print("=" * 70)

df['ey'] = pd.Categorical(df['event_year_bin'])
formula = ('log_revenue_per_capita ~ log_population + C(state_abbr) + C(ey) '
           '+ log_dist_nonlegal:C(ey)')
m = smf.ols(formula, data=df).fit(cov_type='HC1')
mc = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': pd.Categorical(df['fips5']).codes})

rows = []
for ey in sorted(df['event_year_bin'].unique()):
    term = f'log_dist_nonlegal:C(ey)[{ey}]'
    if term not in m.params.index:
        continue  # ey==0 is absorbed into the base log_dist_nonlegal term structure
    b, se, p = m.params[term], m.bse[term], m.pvalues[term]
    se_c, p_c = mc.bse[term], mc.pvalues[term]
    label = f'{ey}+' if ey == 9 else str(ey)
    n_this_year = int((df['event_year_bin'] == ey).sum())
    print(f"event year {label:>3s} (n={n_this_year:4d})   b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} "
          f"| se(county-cl)={se_c:.4f} p={p_c:.4f}")
    rows.append(dict(event_year=label, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c,
                      n=n_this_year))

print(f"\nR2={m.rsquared:.3f} n={int(m.nobs)}")
print("\nFull parameter table (event-year level effects + all interactions):")
print(m.params.filter(like='ey').to_string())

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'revenue_border_eventstudy_results.csv'), index=False)
print("\nSaved tables/revenue_border_eventstudy_results.csv")

print("\n" + "=" * 70)
print("SAME EVENT STUDY, EXCLUDING SEDGWICK (outlier robustness)")
print("=" * 70)
df2 = df[df['county_key'] != 'SEDGWICK'].copy()
df2['ey'] = pd.Categorical(df2['event_year_bin'])
m2 = smf.ols(formula, data=df2).fit(cov_type='HC1')
mc2 = smf.ols(formula, data=df2).fit(cov_type='cluster', cov_kwds={'groups': pd.Categorical(df2['fips5']).codes})
rows2 = []
for ey in sorted(df2['event_year_bin'].unique()):
    term = f'log_dist_nonlegal:C(ey)[{ey}]'
    if term not in m2.params.index:
        continue
    b, se, p = m2.params[term], m2.bse[term], m2.pvalues[term]
    se_c, p_c = mc2.bse[term], mc2.pvalues[term]
    label = f'{ey}+' if ey == 9 else str(ey)
    print(f"event year {label:>3s}   b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f}")
    rows2.append(dict(event_year=label, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c))
pd.DataFrame(rows2).to_csv(os.path.join(TABLES, 'revenue_border_eventstudy_results_no_outlier.csv'), index=False)
print("\nSaved tables/revenue_border_eventstudy_results_no_outlier.csv")
