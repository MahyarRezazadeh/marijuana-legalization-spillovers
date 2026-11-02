# ============================================================
# Two mechanism checks for the revenue-per-store border premium
# (analyze_revenue_perstore_border70.py), both aimed at distinguishing
# "genuine cross-border demand absorption" from "border counties just
# differ for some other reason":
#
# (1) DOSE-RESPONSE: does revenue per store rise smoothly as distance
#     to the nearest non-legal-state county shrinks (continuous), not
#     just jump at an arbitrary 70-mile cutoff? A genuine distance-decay
#     shopping-trip effect should show a smooth gradient; an unrelated
#     confound specific to "being a border county" as a category would
#     not necessarily.
#
# (2) CAPTIVE POPULATION: for each CO/WA county, the SIZE of the demand
#     pool available to absorb matters, not just distance to the
#     nearest non-legal county. A county next to one small non-legal
#     county and a county next to a large non-legal metro both count as
#     "within 70 miles" under the binary/nearest-distance measures, but
#     only the latter has a large captive market to draw from. We
#     compute, for each CO/WA county, the total population of ALL
#     non-legal-state counties within 70 miles (not just the nearest
#     one) as a market-size measure, and test whether revenue per store
#     rises with it.
# ============================================================
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
EARTH_RADIUS_MI = 3958.8
REF_DATE = pd.Timestamp('2026-08-05')

panel = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel.csv'), dtype={'fips5': str})
panel['fips5'] = panel['fips5'].str.zfill(5)

# ---------- (1) Dose-response: continuous distance ----------
print("=" * 70)
print("(1) DOSE-RESPONSE -- outcome: log(revenue per store)")
print("=" * 70)
formula = 'log_revenue_per_store ~ log_dist_nonlegal + log_population + C(fy_year) + C(state_abbr)'
m = smf.ols(formula, data=panel).fit(cov_type='HC1')
mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['county_g']})
b, se, p = m.params['log_dist_nonlegal'], m.bse['log_dist_nonlegal'], m.pvalues['log_dist_nonlegal']
se_c, p_c = mc.bse['log_dist_nonlegal'], mc.pvalues['log_dist_nonlegal']
print(f"{'log(distance to non-legal border) + full controls':52s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} "
      f"| se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
print("(expect NEGATIVE if closer-to-border -> higher revenue per store, i.e. a genuine distance-decay effect)")

dose_row = dict(spec='log(distance to non-legal border) + full controls', coef=b, se_hc1=se, p_hc1=p,
                 se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs))

sub = panel[panel['county_key'] != 'SEDGWICK']
mc2 = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': pd.Categorical(sub['fips5']).codes})
m2 = smf.ols(formula, data=sub).fit(cov_type='HC1')
b2, se2, p2 = m2.params['log_dist_nonlegal'], m2.bse['log_dist_nonlegal'], m2.pvalues['log_dist_nonlegal']
se2_c, p2_c = mc2.bse['log_dist_nonlegal'], mc2.pvalues['log_dist_nonlegal']
print(f"{'[excl. Sedgwick] same spec':52s} b={b2:+.4f} se(HC1)={se2:.4f} p={p2:.4f} "
      f"| se(county-cl)={se2_c:.4f} p={p2_c:.4f} n={int(m2.nobs)}")
dose_row2 = dict(spec='[excl. Sedgwick] log(distance) + full controls', coef=b2, se_hc1=se2, p_hc1=p2,
                  se_county_cl=se2_c, p_county_cl=p2_c, n=int(m2.nobs))

pd.DataFrame([dose_row, dose_row2]).to_csv(os.path.join(TABLES, 'revenue_perstore_doseresponse_results.csv'), index=False)

# ---------- (2) Captive non-legal population within 70mi ----------
print("\n" + "=" * 70)
print("(2) CAPTIVE POPULATION -- total population of non-legal-state")
print("counties within 70mi of each CO/WA county (not just nearest)")
print("=" * 70)

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
legal_states_asof = set(legal.loc[legal['retail_date'].notna() & (legal['retail_date'] <= REF_DATE), 'state_abbr'])

centroids = pd.read_stata(os.path.join(DATA, 'county_centroids.dta'))
centroids['fips5'] = centroids['fips5'].astype(str).str.zfill(5)
pop = pd.read_csv(os.path.join(DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop = pop[pop['SUMLEV'] == 50].copy()
pop['fips5'] = pop['STATE'].astype(str).str.zfill(2) + pop['COUNTY'].astype(str).str.zfill(3)
pop = pop[['fips5', 'POPESTIMATE2023']].rename(columns={'POPESTIMATE2023': 'nl_population'})
centroids = centroids.merge(pop, on='fips5', how='left')

nonlegal = centroids[~centroids['state_abbr'].isin(legal_states_asof)].dropna(subset=['nl_population']).copy()
print(f"Non-legal county universe: {len(nonlegal)} counties, total population {nonlegal['nl_population'].sum():,.0f}")

nonlegal_rad = np.radians(nonlegal[['lat', 'lon']].to_numpy())
tree = BallTree(nonlegal_rad, metric='haversine')
radius_rad = 70 / EARTH_RADIUS_MI

co_wa_counties = panel[['fips5']].drop_duplicates().merge(
    centroids[['fips5', 'lat', 'lon', 'state_abbr']], on='fips5', how='left')
own_rad = np.radians(co_wa_counties[['lat', 'lon']].to_numpy())
indices = tree.query_radius(own_rad, r=radius_rad)

captive_pop = []
for idx_list in indices:
    captive_pop.append(nonlegal.iloc[idx_list]['nl_population'].sum())
co_wa_counties['captive_nonlegal_pop_70mi'] = captive_pop
co_wa_counties['log_captive_pop'] = np.log(co_wa_counties['captive_nonlegal_pop_70mi'] + 1)

print("\nCaptive non-legal population within 70mi, CO/WA counties with any (>0):")
print(co_wa_counties[co_wa_counties['captive_nonlegal_pop_70mi'] > 0]
      .sort_values('captive_nonlegal_pop_70mi', ascending=False)
      [['fips5', 'state_abbr', 'captive_nonlegal_pop_70mi']].to_string(index=False))

panel2 = panel.merge(co_wa_counties[['fips5', 'captive_nonlegal_pop_70mi', 'log_captive_pop']], on='fips5', how='left')
panel2['g2'] = pd.Categorical(panel2['fips5']).codes

print(f"\nCounties with zero captive non-legal population within 70mi: "
      f"{(co_wa_counties['captive_nonlegal_pop_70mi'] == 0).sum()} of {len(co_wa_counties)}")

formula2 = 'log_revenue_per_store ~ log_captive_pop + log_population + C(fy_year) + C(state_abbr)'
m3 = smf.ols(formula2, data=panel2).fit(cov_type='HC1')
mc3 = smf.ols(formula2, data=panel2).fit(cov_type='cluster', cov_kwds={'groups': panel2['g2']})
b3, se3, p3 = m3.params['log_captive_pop'], m3.bse['log_captive_pop'], m3.pvalues['log_captive_pop']
se3_c, p3_c = mc3.bse['log_captive_pop'], mc3.pvalues['log_captive_pop']
print(f"\n{'log(captive non-legal population, 70mi) + full controls':52s} b={b3:+.4f} se(HC1)={se3:.4f} p={p3:.4f} "
      f"| se(county-cl)={se3_c:.4f} p={p3_c:.4f} n={int(m3.nobs)}")
print("(expect POSITIVE if a bigger nearby captive market -> more revenue per store)")

sub2 = panel2[panel2['county_key'] != 'SEDGWICK']
m4 = smf.ols(formula2, data=sub2).fit(cov_type='HC1')
mc4 = smf.ols(formula2, data=sub2).fit(cov_type='cluster', cov_kwds={'groups': pd.Categorical(sub2['fips5']).codes})
b4, se4, p4 = m4.params['log_captive_pop'], m4.bse['log_captive_pop'], m4.pvalues['log_captive_pop']
se4_c, p4_c = mc4.bse['log_captive_pop'], mc4.pvalues['log_captive_pop']
print(f"{'[excl. Sedgwick] same spec':52s} b={b4:+.4f} se(HC1)={se4:.4f} p={p4:.4f} "
      f"| se(county-cl)={se4_c:.4f} p={p4_c:.4f} n={int(m4.nobs)}")

rows = [
    dict(spec='log(captive non-legal population, 70mi) + full controls', coef=b3, se_hc1=se3, p_hc1=p3,
         se_county_cl=se3_c, p_county_cl=p3_c, n=int(m3.nobs)),
    dict(spec='[excl. Sedgwick] log(captive population) + full controls', coef=b4, se_hc1=se4, p_hc1=p4,
         se_county_cl=se4_c, p_county_cl=p4_c, n=int(m4.nobs)),
]
pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'revenue_perstore_captivepop_results.csv'), index=False)
panel2.to_csv(os.path.join(TABLES, 'revenue_perstore_captivepop_panel.csv'), index=False)
print(f"\nSaved to {TABLES}")
