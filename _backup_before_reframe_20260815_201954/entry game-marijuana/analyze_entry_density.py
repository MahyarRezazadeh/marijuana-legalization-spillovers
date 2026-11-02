# ============================================================
# First reduced-form pass: does dispensary density (per 1,000 pop)
# within the 9 dispensary-covered legal states rise as counties get
# closer to a non-legal-state border (a captive illegal-market demand
# pool), controlling for population/land area/state?
#
# Cross-sectional design (one county-level observation each, current
# stock as of 2026-08-05), NOT a stacked-transition panel like the
# rest of the project -- state FE plays the role "transition FE" plays
# elsewhere (absorbs state-level regulatory/market differences), since
# the comparison here is border vs. interior WITHIN the same state.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
retail_map = dict(zip(legal['state_abbr'], legal['retail_date']))

df = pd.read_csv(os.path.join(DATA, 'county_panel_entry.csv'), dtype={'fips5': str})
df['log_dist_nonlegal'] = np.log(df['dist_nonlegal_mi'] + 1)
df['log_population'] = np.log(df['population'])
df['log_land_area'] = np.log(df['land_area_sqmi'])
df['border50'] = (df['dist_nonlegal_mi'] <= 50).astype(int)
df['years_live'] = (pd.Timestamp('2026-08-05') - df['state_abbr'].map(retail_map)).dt.days / 365.25
df['g'] = pd.Categorical(df['state_abbr']).codes

print("=" * 70)
print("UNCONDITIONAL COMPARISON: border (<=50mi of non-legal state) vs interior")
print("=" * 70)
print(df.groupby('border50')['dispensary_per_1000'].agg(['mean', 'median', 'std', 'count']))
print(f"\nBy state (mean dispensary_per_1000, border50 vs interior):")
print(df.groupby(['state_abbr', 'border50'])['dispensary_per_1000'].mean().unstack().round(3))

print("\n" + "=" * 70)
print("REGRESSIONS -- outcome: dispensary_per_1000")
print("=" * 70)

specs = {
    'No controls':                    'dispensary_per_1000 ~ log_dist_nonlegal',
    '+ log(population)':              'dispensary_per_1000 ~ log_dist_nonlegal + log_population',
    '+ state FE':                     'dispensary_per_1000 ~ log_dist_nonlegal + C(state_abbr)',
    '+ log(pop) + state FE':          'dispensary_per_1000 ~ log_dist_nonlegal + log_population + C(state_abbr)',
    '+ log(pop) + land area + state FE': 'dispensary_per_1000 ~ log_dist_nonlegal + log_population + log_land_area + C(state_abbr)',
    '+ log(pop) + land area + years_live + state FE': 'dispensary_per_1000 ~ log_dist_nonlegal + log_population + log_land_area + years_live + C(state_abbr)',
}

rows = []
for name, formula in specs.items():
    m = smf.ols(formula, data=df).fit(cov_type='HC1')
    mc = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['g']})
    b, se, p = m.params['log_dist_nonlegal'], m.bse['log_dist_nonlegal'], m.pvalues['log_dist_nonlegal']
    se_c, p_c = mc.bse['log_dist_nonlegal'], mc.pvalues['log_dist_nonlegal']
    print(f"{name:52s} b={b:+.5f} se(HC1)={se:.5f} p={p:.4f} | se(9-state-cl)={se_c:.5f} p={p_c:.4f} "
          f"R2={m.rsquared:.3f} n={int(m.nobs)}")
    rows.append(dict(spec=name, coef=b, se_hc1=se, p_hc1=p, se_state_cl=se_c, p_state_cl=p_c,
                      r2=m.rsquared, n=int(m.nobs)))

print("\n" + "=" * 70)
print("BINARY VERSION -- outcome: dispensary_per_1000 ~ border50 (<=50mi of non-legal state)")
print("=" * 70)
specs_bin = {
    'No controls':           'dispensary_per_1000 ~ border50',
    '+ log(pop) + state FE': 'dispensary_per_1000 ~ border50 + log_population + C(state_abbr)',
    '+ log(pop) + land area + years_live + state FE':
        'dispensary_per_1000 ~ border50 + log_population + log_land_area + years_live + C(state_abbr)',
}
for name, formula in specs_bin.items():
    m = smf.ols(formula, data=df).fit(cov_type='HC1')
    mc = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['g']})
    b, se, p = m.params['border50'], m.bse['border50'], m.pvalues['border50']
    se_c, p_c = mc.bse['border50'], mc.pvalues['border50']
    print(f"{name:52s} b={b:+.5f} se(HC1)={se:.5f} p={p:.4f} | se(9-state-cl)={se_c:.5f} p={p_c:.4f} "
          f"R2={m.rsquared:.3f} n={int(m.nobs)}")
    rows.append(dict(spec=f'[binary] {name}', coef=b, se_hc1=se, p_hc1=p, se_state_cl=se_c, p_state_cl=p_c,
                      r2=m.rsquared, n=int(m.nobs)))

print("\n" + "=" * 70)
print("ROBUSTNESS: restricted to 7 states with real non-legal-border variation")
print("(drops CA, OR -- both mostly surrounded by other legal states,")
print(" min distance to a non-legal state 222mi/47mi vs 11-25mi for the rest)")
print("=" * 70)
sub = df[~df['state_abbr'].isin(['CA', 'OR'])].copy()
sub['g'] = pd.Categorical(sub['state_abbr']).codes
for name, formula in {
    'No controls': 'dispensary_per_1000 ~ log_dist_nonlegal',
    '+ log(pop) + state FE': 'dispensary_per_1000 ~ log_dist_nonlegal + log_population + C(state_abbr)',
}.items():
    m = smf.ols(formula, data=sub).fit(cov_type='HC1')
    mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
    b, se, p = m.params['log_dist_nonlegal'], m.bse['log_dist_nonlegal'], m.pvalues['log_dist_nonlegal']
    se_c, p_c = mc.bse['log_dist_nonlegal'], mc.pvalues['log_dist_nonlegal']
    print(f"{name:52s} b={b:+.5f} se(HC1)={se:.5f} p={p:.4f} | se(7-state-cl)={se_c:.5f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(spec=f'[7-state, drop CA/OR] {name}', coef=b, se_hc1=se, p_hc1=p, se_state_cl=se_c, p_state_cl=p_c,
                      r2=m.rsquared, n=int(m.nobs)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'entry_density_results.csv'), index=False)
print("\nSaved tables/entry_density_results.csv")
