# ============================================================
# Political-economy test: are Colorado's BORDER counties less likely
# to have effectively excluded retail marijuana (via local bans under
# Amendment 64's opt-out authority, or simply no operator ever
# entering) than interior counties? If local governments recognize
# the fiscal upside of cross-border demand, border counties should
# opt in / allow retail at a higher rate.
#
# IMPORTANT LIMITATION, stated here and carried into the writeup:
# we do not have actual municipal-ordinance ban data (CO's opt-out
# authority is exercised at the city/town level, not uniformly by
# county, and no compiled statewide dataset of ordinance status was
# found). The proxy used here is "did this county ever have an
# operating dispensary, 2015-2025" -- zero dispensaries could mean an
# active local ban, no operator choosing to enter despite it being
# legal, or a small/rural county with no viable market. This proxy
# is directionally informative but conflates "banned" with "never
# entered for any reason" -- reported honestly as a limitation.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

# ---------- Full 64-county CO universe (all counties, not just those with a dispensary) ----------
universe = pd.read_csv(os.path.join(DATA, 'county_universe_with_distance.csv'), dtype={'fips5': str})
co_universe = universe[universe['state_abbr'] == 'CO'].copy()
print(f"Colorado counties, full universe: {len(co_universe)}")

# ---------- Which counties ever had >=1 operating dispensary (2015-2025) ----------
panel = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel_CORRECTED.csv'), dtype={'fips5': str})
co_panel = panel[panel['state_abbr'] == 'CO']
counties_with_dispensary = set(co_panel.loc[co_panel['dispensary_count'] > 0, 'fips5'].unique())
print(f"Colorado counties that ever had an operating dispensary: {len(counties_with_dispensary)}")

co_universe['has_dispensary'] = co_universe['fips5'].isin(counties_with_dispensary).astype(int)
co_universe['border_70mi'] = (co_universe['dist_nonlegal_mi'] <= 70).astype(int)
co_universe['log_population'] = np.log(co_universe['population'])

print(f"\nExcluded (never had a dispensary) counties: {(co_universe['has_dispensary']==0).sum()} of {len(co_universe)}")
print(f"Border counties: {co_universe['border_70mi'].sum()}, Interior: {(co_universe['border_70mi']==0).sum()}")

# ---------- Raw rates ----------
print("\nRaw rate of 'has ever had a dispensary', border vs interior:")
print(co_universe.groupby('border_70mi')['has_dispensary'].agg(['mean', 'sum', 'count']))

# ---------- Regression: does border status predict having a dispensary at all? ----------
print("\n" + "=" * 70)
print("LPM: has_dispensary ~ border_70mi (+ log population)")
print("=" * 70)
for name, formula in {
    'No controls': 'has_dispensary ~ border_70mi',
    '+ log(population)': 'has_dispensary ~ border_70mi + log_population',
}.items():
    m = smf.ols(formula, data=co_universe).fit(cov_type='HC1')
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    print(f"{name:25s} b={b:+.4f} se={se:.4f} p={p:.4f} n={int(m.nobs)}")

# ---------- Same test restricted to counties above a minimal population threshold ----------
# (rules out "border counties just happen to be bigger/smaller" as a confound, and drops
# very small counties where a dispensary was never commercially viable regardless of policy)
print("\n" + "=" * 70)
print("Same test, counties with population >= 5,000 only (drops thinnest rural counties)")
print("=" * 70)
sub = co_universe[co_universe['population'] >= 5000]
print(f"n={len(sub)} counties (dropped {len(co_universe)-len(sub)} counties under 5,000 population)")
m = smf.ols('has_dispensary ~ border_70mi + log_population', data=sub).fit(cov_type='HC1')
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
print(f"+ log(population)         b={b:+.4f} se={se:.4f} p={p:.4f} n={int(m.nobs)}")
print(sub.groupby('border_70mi')['has_dispensary'].agg(['mean', 'sum', 'count']))

co_universe.to_csv(os.path.join(TABLES, 'co_local_ban_political_economy.csv'), index=False)
print(f"\nSaved {os.path.join(TABLES, 'co_local_ban_political_economy.csv')}")
