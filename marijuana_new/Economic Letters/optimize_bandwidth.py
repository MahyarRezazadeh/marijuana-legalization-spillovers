# ============================================================
# Finds a data-driven "optimal" radius for the continuous
# specification's sample restriction, using the CCT
# (Calonico-Cattaneo-Titiunik) MSE-optimal bandwidth selector --
# the same algorithm underlying rdrobust/rdbwselect -- applied
# here purely as a bandwidth-choice TOOL, not to reintroduce the
# border-line RDD design.
#
# Running variable: X_i = dist_2023_mi - 50 (how close each
# county's 2023 distance is to the 50-mile switcher/control
# threshold; negative = switcher side, positive = control side).
# Outcome: d_od (the same outcome as the main regression).
# rdbwselect returns the MSE-optimal bandwidth for local-linear
# estimation around X=0 -- i.e., the optimal RADIUS to include
# on each side of the 50-mile threshold.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from rdrobust import rdbwselect, rdrobust

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel_new.csv'))
main = panel[panel['d_od'].notna()].copy()

main['X'] = main['dist_2023_mi'] - 50  # signed distance from the 50mi threshold, as of 2023

bw_out = rdbwselect(y=main['d_od'].to_numpy(), x=main['X'].to_numpy(), c=0,
                     cluster=main['state_abbr'].to_numpy(), kernel='tri', bwselect='mserd')
print("CCT MSE-optimal bandwidth selection:")
print(bw_out.bws)
h_left = float(bw_out.bws.loc['mserd', 'h (left)'])
h_right = float(bw_out.bws.loc['mserd', 'h (right)'])
h_opt = (h_left + h_right) / 2
print(f"\nOptimal bandwidth: left={h_left:.1f}mi, right={h_right:.1f}mi (avg={h_opt:.1f}mi)")

# Apply this optimal radius as a SAMPLE RESTRICTION to our actual continuous
# specification (d_od ~ d_dist_mi), not the rdrobust local-linear-in-X model --
# i.e., use the algorithm's bandwidth choice, but keep our own regression.
def cluster_fit(y, X, groups):
    Xc = sm.add_constant(X)
    g = pd.Categorical(groups).codes
    return sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})

for label, radius in [('left bw', h_left), ('right bw', h_right), ('avg bw', h_opt), ('rounded', round(h_opt / 5) * 5)]:
    sub = main[(main['dist_2020_mi'] <= 50 + radius) | (main['dist_2023_mi'] <= 50 + radius)]
    r = cluster_fit(sub['d_od'], sub[['d_dist_mi']], sub['state_abbr'])
    n_sw = int((sub['switcher'] == 1).sum()); n_ct = int((sub['switcher'] == 0).sum())
    print(f"{label:10s} (radius from threshold={radius:.1f}mi, total cutoff={50+radius:.0f}mi): "
          f"N={len(sub)} (sw={n_sw}, ct={n_ct}) b={r.params['d_dist_mi']:+.5f} "
          f"se={r.bse['d_dist_mi']:.5f} p={r.pvalues['d_dist_mi']:.4f}")

# Also report rdrobust's own direct estimate at X=0 for reference (this IS a
# proper RDD estimate, shown only as a cross-check on the bandwidth choice,
# not as the paper's specification)
rd_out = rdrobust(y=main['d_od'].to_numpy(), x=main['X'].to_numpy(), c=0,
                   cluster=main['state_abbr'].to_numpy(), kernel='tri', bwselect='mserd')
print(f"\n[Cross-check only] rdrobust local-linear estimate at X=0: "
      f"coef={rd_out.coef.loc['Robust','Coeff']:.4f} se={rd_out.se.loc['Robust','Std. Err.']:.4f} "
      f"p={rd_out.pv.loc['Robust','P>|z|']:.4f}")
