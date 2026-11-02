# ============================================================
# Welfare translation on the new stacked multi-period design:
# applies the pooled switcher coefficient to the total population
# of all distinct switcher counties (summed across their
# respective switch-year transitions), same age/life-expectancy
# approximations as the earlier two-period version.
# ============================================================
import os, json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

with open(os.path.join(DATA, 'multiperiod_results_rec.json')) as f:
    res = json.load(f)
b, se = res['switcher']['coef'], res['switcher']['se']

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
avail = s[s['d_od'].notna() & (s['population'] > 0)]
sw = avail[avail['switcher'] == 1].drop_duplicates('fips5')
pop = sw['population'].sum()
n_counties = len(sw)

REMAINING_LE = 33.4  # unchanged approximation (CDC published summary life-expectancy figures, age ~46)

def averted(coef):
    return -coef * pop / 1e5

pt = averted(b)
lo = averted(b + 1.96 * se)  # less-negative coef -> fewer averted deaths
hi = averted(b - 1.96 * se)
yll = pt * REMAINING_LE

print(f"Switcher counties: {n_counties}, total population: {pop:,.0f}")
print(f"Averted deaths (point): {pt:.1f}  [95% CI: {lo:.1f}, {hi:.1f}]")
print(f"Years of life saved: {yll:,.0f}")

out = dict(n_switcher_counties=n_counties, switcher_population=int(pop),
           coef=b, se=se, averted_deaths=pt, averted_lo=lo, averted_hi=hi,
           remaining_le=REMAINING_LE, yll=yll)
with open(os.path.join(DATA, 'welfare_summary_multiperiod.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved welfare_summary_multiperiod.json")
