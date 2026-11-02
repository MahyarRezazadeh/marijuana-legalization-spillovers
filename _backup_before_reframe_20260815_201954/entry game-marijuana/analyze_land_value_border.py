# ============================================================
# CORRECTED land-value design, replacing analyze_land_value_did.py's
# market-access treatment (which was the wrong geography). Treatment
# here matches the entry-game/border-competition design used
# throughout venous4 and in analyze_revenue_border.py: within the 23
# states with an operating recreational retail market as of
# 2026-08-05, TREATMENT counties are those within 70 miles of a
# non-legal-state border (able to absorb captive cross-border demand
# from residents of the neighboring illegal-market state up to ~70
# miles in); CONTROL counties are the interior/"middle" counties of
# the same legal states, more than 70 miles from any non-legal border.
# Both groups are counties in ALREADY-legal states -- this is not a
# legal-vs-illegal comparison, it is border-vs-interior WITHIN legal
# states.
#
# Outcome is farmland asset value per acre (log), from the same
# national NASS Census-of-Agriculture extraction used in the (now
# superseded) market-access version. Since border status is a fixed
# geographic attribute (a county cannot switch from interior to
# border), this is a panel/cross-sectional comparison, not a
# switcher event study -- mirrors analyze_revenue_border.py's own
# structure exactly (state FE + year FE + population control,
# county-clustered SEs), plus an event-study-by-market-age check
# (does the border/interior gap grow or shrink as each state's own
# legal market matures), mirroring analyze_revenue_border_eventstudy.py.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
THRESH = 70

geo = pd.read_csv(os.path.join(DATA, 'county_universe_all23_with_distance.csv'), dtype={'fips5': str})

nass_dir = os.path.join(BASE, '..', 'marijuana_new', 'health outcomes exploration', 'data')
nass = pd.read_csv(os.path.join(nass_dir, 'nass_land_value_per_acre_county.csv'),
                    dtype={'STATE_FIPS_CODE': str, 'COUNTY_ANSI': str})
nass = nass[nass['VALUE'] != '(D)'].copy()
nass['VALUE'] = nass['VALUE'].str.replace(',', '').astype(float)
nass['YEAR'] = nass['YEAR'].astype(int)
nass = nass.dropna(subset=['STATE_FIPS_CODE', 'COUNTY_ANSI'])
nass['fips5'] = nass['STATE_FIPS_CODE'].str.zfill(2) + nass['COUNTY_ANSI'].str.zfill(3)

panel = geo.merge(nass[['fips5', 'YEAR', 'VALUE']], on='fips5', how='inner')
panel = panel.rename(columns={'YEAR': 'year', 'VALUE': 'landvalue'})
panel['log_landvalue'] = np.log(panel['landvalue'])
panel['log_population'] = np.log(panel['population'])
panel['log_land_area'] = np.log(panel['land_area_sqmi'])
panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= THRESH).astype(int)
panel['log_dist_nonlegal'] = np.log(panel['dist_nonlegal_mi'])
panel['g'] = pd.Categorical(panel['fips5']).codes

print(f"Panel: {len(panel)} county-year obs, {panel['fips5'].nunique()} counties, "
      f"{panel['state_abbr'].nunique()} states, years {sorted(panel['year'].unique())}")
print(f"Border (<= {THRESH}mi) counties: {panel.loc[panel['border_70mi']==1,'fips5'].nunique()}, "
      f"interior counties: {panel.loc[panel['border_70mi']==0,'fips5'].nunique()}")
print("\nStates contributing at least one border county:")
print(sorted(panel.loc[panel['border_70mi']==1, 'state_abbr'].unique()))
print("\nStates with NO border county under this threshold (interior-only):")
print(sorted(set(panel['state_abbr'].unique()) - set(panel.loc[panel['border_70mi']==1, 'state_abbr'].unique())))

print("\n" + "=" * 70)
print("UNCONDITIONAL: mean log land value per acre, border vs interior")
print("=" * 70)
print(panel.groupby('border_70mi')['log_landvalue'].agg(['mean', 'median', 'std', 'count']))

print("\n" + "=" * 70)
print("MAIN PANEL REGRESSIONS -- outcome: log(land value per acre)")
print("(mirrors analyze_revenue_border.py's spec ladder exactly)")
print("=" * 70)
specs = {
    'No controls':                              'log_landvalue ~ border_70mi',
    '+ log(pop)':                                'log_landvalue ~ border_70mi + log_population',
    '+ year FE':                                 'log_landvalue ~ border_70mi + C(year)',
    '+ state FE':                                'log_landvalue ~ border_70mi + C(state_abbr)',
    '+ log(pop) + year FE + state FE':           'log_landvalue ~ border_70mi + log_population + C(year) + C(state_abbr)',
    '+ log(pop) + land area + year FE + state FE': 'log_landvalue ~ border_70mi + log_population + log_land_area + C(year) + C(state_abbr)',
}
rows = []
for name, formula in specs.items():
    m = smf.ols(formula, data=panel).fit(cov_type='HC1')
    mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:48s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} "
          f"R2={m.rsquared:.3f} n={int(m.nobs)}")
    rows.append(dict(spec=name, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, r2=m.rsquared, n=int(m.nobs)))

print("\n" + "=" * 70)
print("CONTINUOUS DISTANCE VERSION -- outcome: log(land value per acre)")
print("=" * 70)
formula = 'log_landvalue ~ log_dist_nonlegal + log_population + log_land_area + C(year) + C(state_abbr)'
m = smf.ols(formula, data=panel).fit(cov_type='HC1')
mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['g']})
b, se, p = m.params['log_dist_nonlegal'], m.bse['log_dist_nonlegal'], m.pvalues['log_dist_nonlegal']
se_c, p_c = mc.bse['log_dist_nonlegal'], mc.pvalues['log_dist_nonlegal']
print(f"{'+ log(pop) + land area + year FE + state FE':48s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} "
      f"| se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
rows.append(dict(spec='[continuous log-distance] + full controls', coef=b, se_hc1=se, p_hc1=p,
                  se_county_cl=se_c, p_county_cl=p_c, r2=m.rsquared, n=int(m.nobs)))

print("\n" + "=" * 70)
print("ROBUSTNESS: restricted to states with real border variation")
print("(drops RI, CT, NV, AZ, CA, AK -- no county within 70mi of a non-legal border)")
print("=" * 70)
weak_states = ['RI', 'CT', 'NV', 'AZ', 'CA', 'AK']
sub = panel[~panel['state_abbr'].isin(weak_states)].copy()
sub['g'] = pd.Categorical(sub['fips5']).codes
formula = 'log_landvalue ~ border_70mi + log_population + log_land_area + C(year) + C(state_abbr)'
m = smf.ols(formula, data=sub).fit(cov_type='HC1')
mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
print(f"{'17-state subsample, full controls':48s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} "
      f"| se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)} n_states={sub['state_abbr'].nunique()}")
rows.append(dict(spec='[17-state subsample, drop weak-border states] full controls', coef=b, se_hc1=se, p_hc1=p,
                  se_county_cl=se_c, p_county_cl=p_c, r2=m.rsquared, n=int(m.nobs)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'land_value_border_results.csv'), index=False)
panel.to_csv(os.path.join(TABLES, 'land_value_border_panel.csv'), index=False)
print(f"\nSaved to {TABLES}")

print("\n" + "=" * 70)
print("EVENT STUDY BY MARKET AGE -- does the border/interior gap change as")
print("each state's own legal market matures? (mirrors analyze_revenue_")
print("border_eventstudy.py). event_year = NASS census year - state's own")
print("retail-launch year, binned")
print("=" * 70)
legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
launch_year = dict(zip(legal['state_abbr'], legal['retail_date'].dt.year))
panel['launch_year'] = panel['state_abbr'].map(launch_year)
panel['event_year'] = panel['year'] - panel['launch_year']
# NASS's 2012/2017/2022 waves predate several states' own legalization (e.g. Delaware's 2025 launch means
# ALL THREE waves are pre-launch for DE; Illinois launched 2020, so 2012/2017 are pre-launch there too) --
# real pre-shock data exists and should not be discarded. Bin: pooled far-pre (<=-4), individual near-pre
# (-3,-2,-1), individual post (0..8), pooled 8+.
panel['event_year_bin'] = panel['event_year'].clip(lower=-4, upper=8)

panel['ey'] = pd.Categorical(panel['event_year_bin'])
formula = 'log_landvalue ~ log_population + log_land_area + C(state_abbr) + C(ey) + border_70mi:C(ey)'
m = smf.ols(formula, data=panel).fit(cov_type='HC1')
g2 = pd.Categorical(panel['fips5']).codes
mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': g2})

rows2 = []
for ey in sorted(panel['event_year_bin'].unique()):
    term = f'border_70mi:C(ey)[{ey}]'
    if term not in m.params.index:
        continue
    b, se, p = m.params[term], m.bse[term], m.pvalues[term]
    se_c, p_c = mc.bse[term], mc.pvalues[term]
    if ey == 8:
        label = '8+'
    elif ey == -4:
        label = '-4 or earlier'
    else:
        label = str(int(ey))
    n_this = int((panel['event_year_bin'] == ey).sum())
    print(f"market-age year {label:>13s} (n={n_this:4d})   b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} "
          f"| se(county-cl)={se_c:.4f} p={p_c:.4f}")
    rows2.append(dict(event_year=label, event_year_numeric=int(ey), coef=b, se_hc1=se, p_hc1=p,
                       se_county_cl=se_c, p_county_cl=p_c, n=n_this))

pd.DataFrame(rows2).to_csv(os.path.join(TABLES, 'land_value_border_eventstudy_results.csv'), index=False)
print(f"\nSaved land_value_border_eventstudy_results.csv")
