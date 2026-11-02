# ============================================================
# Land value event study. Extends analyze_land_value_did.py's 2-period
# DiD into a coarse event-time design: since NASS land values are only
# observed in 2012/2017/2022 while distance/switching is tracked
# annually 2020-2025, the only usable "event time" variation is how
# many years before 2022 a county switched (switch_year takes only the
# values 2022, 2023, or 2024 in this sample -- no county switches in
# 2021 or 2025). Relative to the 2022 outcome measurement:
#   et_0  = switched in 2022 (treated AT the measurement date)
#   et_m1 = switches in 2023 (NOT yet treated as of 2022 -- one year out)
#   et_m2 = switches in 2024 (NOT yet treated as of 2022 -- two years out)
#   control = always-far (never within 50mi at either 2020 or 2025)
# et_m1 and et_m2 serve as internal not-yet-treated placebos, tested
# against the SAME 2017-2022 growth window as et_0. All coefficients
# are estimated jointly against BOTH the pre-period (2012-2017, where
# no county is treated -- the parallel-trends benchmark) and post
# period (2017-2022) in one stacked regression, following the DiD logic
# of Table 3 in Schwab & Butsic (2020), county-clustered SEs.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, '..', 'Economic Letters', 'data')
OUT = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\entry game-marijuana"
TABLES = os.path.join(OUT, 'tables')
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

df['g_pre'] = np.log(df['landval_2017']) - np.log(df['landval_2012'])
df['g_post'] = np.log(df['landval_2022']) - np.log(df['landval_2017'])


def event_bin(sy):
    if pd.isna(sy):
        return 'control'
    return {2022: 'et_0', 2023: 'et_m1', 2024: 'et_m2'}.get(int(sy), 'other')


df['event_bin'] = df['switch_year'].apply(event_bin)
df = df[df['event_bin'] != 'other']
print("Event-bin sizes:")
print(df['event_bin'].value_counts())

long = pd.concat([
    df.assign(period='pre', growth=df['g_pre'], post=0),
    df.assign(period='post', growth=df['g_post'], post=1),
], ignore_index=True)
long['g'] = pd.Categorical(long['fips5']).codes
long['eb'] = pd.Categorical(long['event_bin'], categories=['control', 'et_m2', 'et_m1', 'et_0'])

for b in ['et_m2', 'et_m1', 'et_0']:
    long[b] = (long['event_bin'] == b).astype(int)
    long[f'post_x_{b}'] = long['post'] * long[b]

formula = 'growth ~ post + et_m2 + et_m1 + et_0 + post_x_et_m2 + post_x_et_m1 + post_x_et_0 + log_population'
m = smf.ols(formula, data=long).fit(cov_type='cluster', cov_kwds={'groups': long['g']})

print("\n" + "=" * 70)
print("EVENT-STUDY COEFFICIENTS (interaction = post x event-time bin)")
print("outcome: log growth in land value per acre; ref = always-far control")
print("=" * 70)
rows = []
labels = {'post_x_et_m2': 'et_m2 (switches 2024, not-yet-treated by 2022, 2yr out)',
          'post_x_et_m1': 'et_m1 (switches 2023, not-yet-treated by 2022, 1yr out)',
          'post_x_et_0': 'et_0  (switches 2022, treated at measurement date)'}
for term, label in labels.items():
    b, se, p = m.params[term], m.bse[term], m.pvalues[term]
    print(f"{label:58s} b={b:+.4f} se(county-cl)={se:.4f} p={p:.4f}")
    rows.append(dict(event_time=term, label=label, coef=b, se=se, p=p))
print(f"\nR2={m.rsquared:.3f} n={int(m.nobs)} n_counties={df['fips5'].nunique()}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'land_value_eventstudy_results.csv'), index=False)
df.to_csv(os.path.join(TABLES, 'land_value_eventstudy_panel.csv'), index=False)
print("\nSaved to", TABLES)
