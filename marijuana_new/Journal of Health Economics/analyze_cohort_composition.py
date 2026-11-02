# ============================================================
# Mechanism check #3: does the post-switch "rebound" (positive
# coefficients at t=+1..+3 in the pooled event study) show up
# WITHIN individual switch-year cohorts, or is it an artifact of
# pooling different cohorts (different real calendar years,
# different neighboring states) into the same event-time bucket?
#
# Three cohorts exist in the main design: switch_year 2022 (n=39,
# observable et=-1..+3), 2023 (n=248, et=-2..+2), 2024 (n=149,
# et=-3..+1) -- each cohort's real event-time window is fixed by
# where its switch year falls in the 2020-2025 panel, so this is a
# genuine within-cohort test, not just a relabeling.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

panel = pd.read_csv(os.path.join(DATA, 'eventstudy_main_panel.csv'), dtype={'fips5': str})

# recover each switcher county's switch_year from its et_0 row's transition
et0 = panel[panel['bucket'] == 'et_0'][['fips5', 'transition']].copy()
et0['switch_year'] = et0['transition'].str.split('-').str[1].astype(int)
switch_year_map = et0.set_index('fips5')['switch_year'].to_dict()

panel['switch_year'] = panel['fips5'].map(switch_year_map)  # NaN for never-switchers (controls)

EVENT_BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']

results = {}
for cohort in [2022, 2023, 2024]:
    cohort_fips = set(et0.loc[et0['switch_year'] == cohort, 'fips5'])
    # sample: this cohort's switcher rows (all event-time buckets) + all never-switcher control rows
    sub = panel[(panel['fips5'].isin(cohort_fips)) | (panel['bucket'] == 'control')].copy()
    sub = sub.dropna(subset=['baseline_rate'])
    sub['county_g'] = pd.Categorical(sub['fips5']).codes

    present_bins = [b for b in EVENT_BINS if sub[b].sum() > 0]
    rhs = ' + '.join(present_bins)
    m = smf.ols(f'd_od ~ {rhs} + baseline_rate + C(transition)', data=sub).fit(
        cov_type='cluster', cov_kwds={'groups': sub['county_g']})

    print(f"\n=== Cohort switch_year={cohort} (n={len(cohort_fips)} counties) ===")
    cohort_out = {}
    for b in present_bins:
        n_b = int((sub['bucket'] == b).sum())
        cohort_out[b] = dict(coef=float(m.params[b]), se=float(m.bse[b]), p=float(m.pvalues[b]), n=n_b)
        print(f"  {b}: b={m.params[b]:+.4f} se={m.bse[b]:.4f} p={m.pvalues[b]:.4f}  n={n_b}")
    results[cohort] = cohort_out

with open(os.path.join(DATA, 'cohort_composition_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved cohort_composition_results.json")

print("\n\n=== SUMMARY: t=+1 coefficient by cohort (the key test) ===")
for cohort, out in results.items():
    if 'et_p1' in out:
        r = out['et_p1']
        print(f"  {cohort}: b={r['coef']:+.4f} p={r['p']:.4f} n={r['n']}")
    else:
        print(f"  {cohort}: t+1 not observable in panel window")
