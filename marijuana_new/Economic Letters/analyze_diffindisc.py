# ============================================================
# Appendix robustness check: a genuine diff-in-discontinuities
# design, using the SAME distance-to-nearest-legal-market-county
# variable already computed for the main design (not a separately
# reconstructed border-line geometry, which is what a prior spatial
# RDD attempt on the two-snapshot design used and found null).
#
# Running variable: distance to nearest legal-market county, each
# year. Cutoff: the substantive 50-mile threshold used throughout
# this paper. Bandwidth: NOT fixed -- chosen by the CCT
# (Calonico-Cattaneo-Titiunik) MSE-optimal bandwidth selector
# separately in each cross-section, so the "optimum distance" is
# data-driven rather than assumed.
#
# Diff-in-disc = RD jump in 2025 (the year with the most legal-
# market states, and so the most data density near the 50-mile
# margin) minus RD jump in 2020 (the baseline year, thinnest data
# near the margin) -- the Grembi-Nannicini-Troiano approach, same
# as the prior spatial-RDD attempt on this project, just with a
# simpler and more directly comparable running variable.
# ============================================================
import os, json, math
import numpy as np
import pandas as pd
from rdrobust import rdbwselect, rdrobust

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

CUTOFF = 50
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

dist = pd.read_csv(os.path.join(DATA, 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
with open(os.path.join(DATA, 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    od_raw = pd.DataFrame(json.load(f))
od_raw['fips5'] = od_raw['fips'].astype(str).str.zfill(5)
od_raw['year'] = od_raw['year'].astype(int)
od_raw['provisional_drug_overdose'] = pd.to_numeric(od_raw['provisional_drug_overdose'], errors='coerce')
od = od_raw.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od.columns = [f'od_{c}' for c in od.columns]
od = od.reset_index()

panel = dist.merge(od, on='fips5', how='left')

def run_rd(year):
    sub = panel[panel['population'] > 0].copy()
    sub['od_rate'] = sub[f'od_{year}'] / (sub['population'] / 1e5)
    sub['X'] = sub[f'dist_{year}'] - CUTOFF
    sub = sub[sub['od_rate'].notna() & sub['X'].notna()]

    bw_out = rdbwselect(y=sub['od_rate'].to_numpy(), x=sub['X'].to_numpy(), c=0,
                         cluster=sub['state_abbr'].to_numpy(), kernel='tri', bwselect='mserd')
    h_left = float(bw_out.bws.loc['mserd', 'h (left)'])
    h_right = float(bw_out.bws.loc['mserd', 'h (right)'])

    rd_out = rdrobust(y=sub['od_rate'].to_numpy(), x=sub['X'].to_numpy(), c=0,
                       cluster=sub['state_abbr'].to_numpy(), kernel='tri', bwselect='mserd')
    coef = float(rd_out.coef.loc['Robust', 'Coeff'])
    se = float(rd_out.se.loc['Robust', 'Std. Err.'])
    p = float(rd_out.pv.loc['Robust', 'P>|z|'])
    n_eff = int(rd_out.N_h[0] + rd_out.N_h[1])
    return dict(year=year, h_left=h_left, h_right=h_right, coef=coef, se=se, p=p, n_eff=n_eff, n_total=len(sub))

results = [run_rd(y) for y in YEARS]
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

r2020 = results[0]
r2025 = results[-1]
diff = r2025['coef'] - r2020['coef']
diff_se = np.sqrt(r2025['se']**2 + r2020['se']**2)  # independent cross-sections
diff_z = diff / diff_se
diff_p = 2 * (1 - 0.5 * (1 + math.erf(abs(diff_z) / np.sqrt(2))))
print(f"\nDiff-in-disc (2025 - 2020): {diff:+.4f} (se={diff_se:.4f}, z={diff_z:.3f}, p={diff_p:.4f})")
print(f"2020 RD: bw=[{r2020['h_left']:.1f}, {r2020['h_right']:.1f}]mi, n_eff={r2020['n_eff']}")
print(f"2025 RD: bw=[{r2025['h_left']:.1f}, {r2025['h_right']:.1f}]mi, n_eff={r2025['n_eff']}")

res_df.to_csv(os.path.join(TABLES, 'diffindisc_by_year.csv'), index=False)
out = dict(by_year=results, diff=diff, diff_se=diff_se, diff_z=diff_z, diff_p=diff_p)
with open(os.path.join(DATA, 'diffindisc_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved diffindisc_by_year.csv, diffindisc_results.json")
