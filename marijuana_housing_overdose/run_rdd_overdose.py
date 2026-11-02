# ============================================================
# Spatial RDD / diff-in-discontinuities for overdose mortality.
# Running variable = SIGNED distance to the actual state-border
# line (negative = never-legal side, positive = legal-market
# side), cutoff = 0. Two approaches:
#   (1) rdrobust (Calonico-Cattaneo-Titiunik) run separately on
#       the 2020 and 2023 cross-sections -- gives a genuinely
#       data-driven ("optimal") bandwidth per period, robust
#       bias-corrected inference, clustered by state. The
#       diff-in-disc estimate is the difference of the two jumps.
#   (2) A standard pooled local-linear diff-in-disc regression
#       (Grembi-Nannicini-Troiano style) at a fixed ~50mi common
#       bandwidth, clustered SE by state, as the user requested.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from rdrobust import rdrobust

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

# ── Border distances (unsigned, from feasibility_rdd.py) ──
dist_df = pd.read_csv(os.path.join(DATA, 'rdd_border_distances.csv'), dtype={'fips5': str})
dist_df['X'] = np.where(dist_df['side'] == 'legal', dist_df['dist_mi'], -dist_df['dist_mi'])

# ── Population (all US counties, vintage 2020/2023 estimates) ──
pop = pd.read_csv(os.path.join(DATA, 'co_est2023_alldata.csv'), encoding='latin1')
pop = pop[pop['COUNTY'] != 0].copy()
pop['fips5'] = pop['STATE'].astype(str).str.zfill(2) + pop['COUNTY'].astype(str).str.zfill(3)
pop20 = pop[['fips5', 'POPESTIMATE2020']].rename(columns={'POPESTIMATE2020': 'pop'})
pop20['year'] = 2020
pop23 = pop[['fips5', 'POPESTIMATE2023']].rename(columns={'POPESTIMATE2023': 'pop'})
pop23['year'] = 2023
pop_long = pd.concat([pop20, pop23], ignore_index=True)

# ── Overdose raw counts (nationwide pull, both years) ──
with open(os.path.join(DATA, 'overdose_county_raw.json'), encoding='utf-8-sig') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od_raw['year'] = od_raw['year'].astype(int)

panel = dist_df.merge(pop_long, on=['fips5', 'year'], how='left')
panel = panel.merge(od_raw[['fips5', 'year', 'provisional_drug_overdose']], on=['fips5', 'year'], how='left')
panel['od_rate'] = panel['provisional_drug_overdose'] / (panel['pop'] / 1e5)
panel.to_csv(os.path.join(DATA, 'rdd_overdose_panel.csv'), index=False)
print(f"RDD panel: {len(panel)} rows. Non-missing outcome: {panel['od_rate'].notna().sum()}")
for yr in [2020, 2023]:
    sub = panel[panel['year'] == yr]
    print(f"  {yr}: N={len(sub)}, non-missing rate={sub['od_rate'].notna().sum()}, "
          f"X range=[{sub['X'].min():.0f}, {sub['X'].max():.0f}]")

# ============================================================
# (1) rdrobust, CCT-optimal bandwidth, separately per period
# ============================================================
print("\n" + "="*70)
print("(1) rdrobust: CCT robust bias-corrected RDD, per period, clustered by state")
print("="*70)

rd_results = {}
for yr in [2020, 2023]:
    sub = panel[(panel['year'] == yr) & panel['od_rate'].notna()].copy()
    out = rdrobust(y=sub['od_rate'].to_numpy(), x=sub['X'].to_numpy(), c=0,
                    cluster=sub['state'].to_numpy(), kernel='tri', bwselect='mserd')
    coef = out.coef.loc['Robust', 'Coeff']
    se = out.se.loc['Robust', 'Std. Err.']
    pval = out.pv.loc['Robust', 'P>|z|']
    h_l, h_r = float(out.bws.loc['h', 'left']), float(out.bws.loc['h', 'right'])
    n_l, n_r = int(out.N_h[0]), int(out.N_h[1])
    rd_results[yr] = dict(coef=coef, se=se, p=pval, h_left=h_l, h_right=h_r, n_left=n_l, n_right=n_r)
    print(f"\n{yr}: jump at border = {coef:+.3f} (se={se:.3f}, p={pval:.3f})")
    print(f"  CCT-optimal bandwidth: {h_l:.1f}mi (never-legal side), {h_r:.1f}mi (legal side)")
    print(f"  Effective N within bandwidth: {n_l} (never-legal side), {n_r} (legal side)")

diff_in_disc = rd_results[2023]['coef'] - rd_results[2020]['coef']
diff_se = np.sqrt(rd_results[2023]['se']**2 + rd_results[2020]['se']**2)
diff_z = diff_in_disc / diff_se
from scipy.stats import norm
diff_p = 2 * (1 - norm.cdf(abs(diff_z)))
print(f"\nDIFF-IN-DISC (CCT-optimal-bandwidth approach): "
      f"{diff_in_disc:+.3f} (se={diff_se:.3f}, p={diff_p:.3f})")
print("  = (2023 border jump) - (2020 border jump), each at its own CCT-optimal bandwidth")

# ============================================================
# (2) Pooled local-linear diff-in-disc at a fixed ~50mi bandwidth
# ============================================================
print("\n" + "="*70)
print("(2) Pooled diff-in-disc regression, common 50mi bandwidth, clustered by state")
print("="*70)

BW = 50
pooled = panel[(panel['X'].abs() <= BW) & panel['od_rate'].notna()].copy()
pooled['Post'] = (pooled['year'] == 2023).astype(int)
pooled['Right'] = (pooled['X'] > 0).astype(int)
pooled['PostRight'] = pooled['Post'] * pooled['Right']
pooled['X_Right'] = pooled['X'] * pooled['Right']
pooled['X_Post'] = pooled['X'] * pooled['Post']
pooled['X_PostRight'] = pooled['X'] * pooled['Post'] * pooled['Right']

Xcols = ['Post', 'Right', 'PostRight', 'X', 'X_Right', 'X_Post', 'X_PostRight']
Xmat = sm.add_constant(pooled[Xcols])
g = pd.Categorical(pooled['state']).codes
res = sm.OLS(pooled['od_rate'], Xmat).fit(cov_type='cluster', cov_kwds={'groups': g})
b, se, p = res.params['PostRight'], res.bse['PostRight'], res.pvalues['PostRight']
n = int(res.nobs)
print(f"\nDiff-in-disc coefficient (Post x Right), 50mi common bandwidth: "
      f"{b:+.3f} (se={se:.3f}, p={p:.3f}), N={n}")
print(res.summary().tables[1])

summary_rows = [
    dict(approach='rdrobust_2020', coef=rd_results[2020]['coef'], se=rd_results[2020]['se'],
         p=rd_results[2020]['p'], bw_left=rd_results[2020]['h_left'], bw_right=rd_results[2020]['h_right'],
         n_left=rd_results[2020]['n_left'], n_right=rd_results[2020]['n_right']),
    dict(approach='rdrobust_2023', coef=rd_results[2023]['coef'], se=rd_results[2023]['se'],
         p=rd_results[2023]['p'], bw_left=rd_results[2023]['h_left'], bw_right=rd_results[2023]['h_right'],
         n_left=rd_results[2023]['n_left'], n_right=rd_results[2023]['n_right']),
    dict(approach='diff_in_disc_optimal_bw', coef=diff_in_disc, se=diff_se, p=diff_p,
         bw_left=np.nan, bw_right=np.nan, n_left=np.nan, n_right=np.nan),
    dict(approach='diff_in_disc_pooled_50mi', coef=b, se=se, p=p,
         bw_left=BW, bw_right=BW, n_left=int((pooled['Right']==0).sum()), n_right=int((pooled['Right']==1).sum())),
]
pd.DataFrame(summary_rows).to_csv(os.path.join(TABLES, 'tF1_rdd_overdose.csv'), index=False)
print(f"\nSaved tF1_rdd_overdose.csv")
