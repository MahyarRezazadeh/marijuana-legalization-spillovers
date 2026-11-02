# ============================================================
# Land value DiD: does gaining access to a legal cannabis market
# (same distance-to-nearest-legal-market-county treatment as the main
# mortality paper) raise farmland asset value per acre?
#
# KEY DESIGN NOTE: this is NOT a true annual event study like the main
# paper's. NASS Census-of-Agriculture land values only exist at three
# points -- 2012, 2017, 2022 -- while the project's distance/switch data
# only runs 2020-2025. So:
#   - g_pre  = log(landval_2017) - log(landval_2012)   [entirely PRE any
#              possible switch, since no switching happens before 2020]
#   - g_post = log(landval_2022) - log(landval_2017)   [spans the actual
#              2020-2025 switch window]
# This gives a legitimate 2-period DiD (parallel-trends check = do
# switcher/control g_pre match?), but treatment timing WITHIN g_post is
# staggered and heterogeneous: a county that switches in 2021 has ~1
# year of "treatment" baked into g_post by the 2022 census; a county
# that switches in 2025 has NONE (2022 predates its switch entirely).
# Following this project's own standing practice of avoiding naive
# staggered-adoption pooling (Goodman-Bacon 2021, already cited in the
# main paper), we do NOT pool all "eventual switchers" into one
# treatment group by default -- we split by whether the switch actually
# happened by 2022, and use the not-yet-treated switchers as an
# additional internal placebo.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, '..', 'Economic Letters', 'data')
TABLES = os.path.join(BASE, 'tables')
os.makedirs(TABLES, exist_ok=True)

dist = pd.read_csv(os.path.join(SRC, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
THRESH = 50


def find_switch_year(row):
    for y in YEARS[1:]:
        if row[f'dist_{y-1}'] >= THRESH and row[f'dist_{y}'] < THRESH:
            return y
    return np.nan


dist = dist[dist['dist_2020'] >= THRESH].copy()
dist['switch_year'] = dist.apply(find_switch_year, axis=1)
dist['log_population'] = np.log(dist['population'])

nass = pd.read_csv(os.path.join(BASE, 'data', 'nass_land_value_per_acre_county.csv'),
                    dtype={'STATE_FIPS_CODE': str, 'COUNTY_ANSI': str})
nass = nass[nass['VALUE'] != '(D)'].copy()
nass['VALUE'] = nass['VALUE'].str.replace(',', '').astype(float)
nass['YEAR'] = nass['YEAR'].astype(int)
nass = nass.dropna(subset=['STATE_FIPS_CODE', 'COUNTY_ANSI'])
nass['fips5'] = nass['STATE_FIPS_CODE'].str.zfill(2) + nass['COUNTY_ANSI'].str.zfill(3)
wide = nass.pivot_table(index='fips5', columns='YEAR', values='VALUE', aggfunc='first')
wide.columns = [f'landval_{c}' for c in wide.columns]
wide = wide.reset_index()

df = dist.merge(wide, on='fips5', how='inner')
df = df.dropna(subset=['landval_2012', 'landval_2017', 'landval_2022'])
print(f"Matched panel: {len(df)} counties with all 3 NASS years present")

df['g_pre'] = np.log(df['landval_2017']) - np.log(df['landval_2012'])
df['g_post'] = np.log(df['landval_2022']) - np.log(df['landval_2017'])

df['group'] = 'always_far'
df.loc[df['switch_year'].notna() & (df['switch_year'] <= 2022), 'group'] = 'switched_by_2022'
df.loc[df['switch_year'].notna() & (df['switch_year'] > 2022), 'group'] = 'switches_after_2022'

print("\nGroup sizes:")
print(df['group'].value_counts())

print("\n" + "=" * 70)
print("STEP 1: PARALLEL-TRENDS CHECK -- pre-period (2012-2017) growth by group")
print("(should be similar across groups if parallel-trends assumption holds;")
print(" this period predates ANY possible switch, so no group is treated yet)")
print("=" * 70)
print(df.groupby('group')['g_pre'].agg(['mean', 'median', 'std', 'count']))


def run_did(sub, label):
    sub = sub.copy()
    long = pd.concat([
        sub.assign(period='pre', growth=sub['g_pre'], post=0),
        sub.assign(period='post', growth=sub['g_post'], post=1),
    ], ignore_index=True)
    long['treat'] = (long['group'] != 'always_far').astype(int)
    long['post_x_treat'] = long['post'] * long['treat']
    long['g'] = pd.Categorical(long['fips5']).codes

    formula = 'growth ~ post + treat + post_x_treat + log_population'
    m = smf.ols(formula, data=long).fit(cov_type='cluster', cov_kwds={'groups': long['g']})
    b, se, p = m.params['post_x_treat'], m.bse['post_x_treat'], m.pvalues['post_x_treat']
    print(f"\n{label}")
    print(f"  DiD (post x treat): b={b:+.4f} se(county-cl)={se:.4f} p={p:.4f} n_counties={sub['fips5'].nunique()}")
    return dict(spec=label, coef=b, se=se, p=p, n=sub['fips5'].nunique())


print("\n" + "=" * 70)
print("STEP 2: DiD ESTIMATES")
print("=" * 70)

rows = []
rows.append(run_did(df[df['group'].isin(['always_far', 'switched_by_2022'])],
                     'MAIN: switched by 2022 (actually treated in g_post) vs always-far'))
rows.append(run_did(df[df['group'].isin(['always_far', 'switches_after_2022'])],
                     'PLACEBO: switches AFTER 2022 (not-yet-treated in g_post) vs always-far -- should be null'))
# intent-to-treat: pool ALL eventual switchers regardless of timing (diluted, for contrast)
itt = df.copy()
itt['group'] = np.where(itt['switch_year'].notna(), 'eventual_switcher', 'always_far')
rows.append(run_did(itt, 'INTENT-TO-TREAT (diluted): ALL eventual switchers 2021-2025 pooled vs always-far'))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'land_value_did_results.csv'), index=False)
print("\nSaved tables/land_value_did_results.csv")
