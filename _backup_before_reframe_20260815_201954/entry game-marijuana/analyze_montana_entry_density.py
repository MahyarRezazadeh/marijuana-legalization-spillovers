# ============================================================
# Direct test, never previously run: does Montana -- the one sample
# state without a statewide dispensary license cap -- actually show
# MORE dispensaries (not just higher per-capita revenue) in border
# counties? Montana was never part of the original 9-state
# entry-density universe (CA/CO/IL/MA/ME/NY/OH/OR/WA), so this
# specific implication of the license-cap interpretation has never
# been checked. If the entry-channel story is right, dispensary
# count/density should be higher in Montana's border counties, unlike
# the null found for the capped CO+WA states.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

mt = pd.read_csv(os.path.join(TABLES, 'montana_revenue_perstore_cross.csv'), dtype={'fips5': str})
mt = mt.drop_duplicates(subset=['fips5'])  # 2025 cross-section, one row per county
mt['dispensary_per_1000'] = mt['dispensary_count'] / mt['population'] * 1000
mt['dispensary_per_sqmi100'] = mt['dispensary_count'] / mt['land_area_sqmi'] * 100
mt['log_dispensary_count'] = np.log(mt['dispensary_count'])

print(f"Montana counties in this cross-section: {len(mt)}")
print(f"Border counties: {mt['border_70mi'].sum()}, Interior: {(mt['border_70mi']==0).sum()}")
print()
print("Raw means, dispensary count/density, border vs interior:")
print(mt.groupby('border_70mi')[['dispensary_count', 'dispensary_per_1000', 'dispensary_per_sqmi100']].mean())
print()

for outcome, label in [('dispensary_count', 'raw count'), ('log_dispensary_count', 'log(count)'),
                        ('dispensary_per_1000', 'per 1,000 residents'), ('dispensary_per_sqmi100', 'per 100 sq mi')]:
    formula = f'{outcome} ~ border_70mi + log_population'
    m = smf.ols(formula, data=mt).fit(cov_type='HC1')
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    print(f"{label:20s} (+pop control): b={b:+.4f} se={se:.4f} p={p:.4f} n={int(m.nobs)}")

print()
formula_nc = 'dispensary_per_1000 ~ border_70mi'
m_nc = smf.ols(formula_nc, data=mt).fit(cov_type='HC1')
print(f"No controls, per-1000: b={m_nc.params['border_70mi']:+.4f} p={m_nc.pvalues['border_70mi']:.4f}")

mt.to_csv(os.path.join(TABLES, 'montana_entry_density_panel.csv'), index=False)
print("\nSaved montana_entry_density_panel.csv")
