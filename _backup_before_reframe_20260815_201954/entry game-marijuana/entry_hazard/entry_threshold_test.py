# ============================================================
# Direct Bresnahan-Reiss-style entry-threshold test: does the local
# POPULATION required to support a 2nd dispensary differ between
# border and interior counties? This is the classic cross-sectional
# threshold-ratio approach (not the hazard/timing approach used
# elsewhere), applied here as an independent second test of the same
# underlying idea: if cross-border demand effectively adds to a
# county's market size, border counties should cross the "support a
# 2nd store" population threshold at a SMALLER local population than
# interior counties need.
#
# Uses the CURRENT (2026-08-05 snapshot) dispensary count per county --
# the same count used in the static entry-density check -- not the
# hazard panel's timed entry-day sequence, since this is a
# cross-sectional threshold question, not a timing question.
# ============================================================
import os
import numpy as np
import pandas as pd

PROJ_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ_BASE, 'data')
OUT = os.path.dirname(os.path.abspath(__file__))

panel = pd.read_csv(os.path.join(DATA, 'county_panel_entry.csv'), dtype={'fips5': str})
panel['border_70mi'] = (panel['dist_nonlegal_mi'] <= 70).astype(int)
panel['n_stores_capped'] = panel['dispensary_count'].clip(upper=4)

print("Distribution of current dispensary count, border vs interior:")
print(pd.crosstab(panel['border_70mi'], panel['n_stores_capped']))

print("\nMedian population by (border status x current store count):")
tab = panel.groupby(['border_70mi', 'n_stores_capped'])['population'].median().unstack(0)
tab.columns = ['interior', 'border']
print(tab)

print("\nMean log(population) by (border status x current store count), with counts:")
for n in sorted(panel['n_stores_capped'].unique()):
    for b, lbl in [(0, 'interior'), (1, 'border')]:
        sub = panel[(panel['n_stores_capped'] == n) & (panel['border_70mi'] == b)]
        if len(sub) == 0:
            continue
        print(f"  N={n} ({lbl}): n_counties={len(sub):3d}, "
              f"median pop={sub['population'].median():>10,.0f}, "
              f"mean log(pop)={np.log(sub['population']).mean():.3f}")

# Entry-threshold ratios: population needed to go from N-1 -> N stores,
# separately for border and interior (Bresnahan-Reiss S_N/S_{N-1} logic,
# approximated here with median population at each count level since
# our county universe is much smaller than B&R's national sawmill/doctor
# samples and a full structural threshold model is not feasible).
print("\nApproximate threshold ratios (median pop at N stores / median pop at N-1 stores):")
for b, lbl in [(0, 'interior'), (1, 'border')]:
    meds = panel[panel['border_70mi'] == b].groupby('n_stores_capped')['population'].median()
    print(f"  {lbl}: {dict(meds.round(0))}")
    for n in [1, 2, 3, 4]:
        if n in meds.index and (n - 1) in meds.index and meds[n - 1] > 0:
            print(f"    S_{n}/S_{n-1} = {meds[n] / meds[n-1]:.2f}")

panel.to_csv(os.path.join(OUT, 'entry_threshold_panel.csv'), index=False)
