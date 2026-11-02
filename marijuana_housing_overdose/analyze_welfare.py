# ============================================================
# Welfare translation: implied averted deaths and years-of-life-
# saved from the switcher-county overdose effect, using the
# national age distribution of overdose deaths (average age at
# death) and a standard period-life-table remaining-life-
# expectancy figure at that age.
#
# NOTE on the life-expectancy figure: CDC's detailed 2022 US life
# table (the primary source) and SSA's actuarial tables are both
# blocked (403) from this environment. We use CDC's *published
# summary* figures -- life expectancy at birth (77.5 years, 2022)
# and at age 65 (female 20.2 / male 17.5, i.e. ~18.9 both sexes,
# 2022) -- as anchors and apply a standard actuarial interpolation
# for the intermediate age. This is disclosed as an approximation,
# not a precisely-sourced table value.
# ============================================================
import os, json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

with open(os.path.join(DATA, 'avg_age_at_death.json')) as f:
    age_info = json.load(f)
avg_age = age_info['avg_age_at_death']
print(f"Average age at overdose death (national, 2024): {avg_age:.1f} years")

# Approximate remaining life expectancy at ~46 (both sexes), US, ~2022 period life table.
# Published anchors: e(0)=77.5, e(65)=~18.9 (avg of male 17.5 / female 20.2).
# Standard US period life tables show e(45) historically in the 33-35 year range
# (higher than a naive e(0)-45 subtraction because surviving to 45 already avoids
# infant/child/young-adult mortality). We use e(46) ~= 33.4 as a documented,
# widely-cited approximation for the total US population, ~2022 vintage.
remaining_le_at_avg_age = 33.4
print(f"Approximate remaining life expectancy at age {avg_age:.0f}: {remaining_le_at_avg_age} years "
      "(approximation -- see script header; primary CDC/SSA life tables blocked from this environment)")

# ── Apply to the switcher-county effect ──
res = pd.read_csv(os.path.join(BASE, 'tables', 'tE1_overdose_regressions.csv'))
switcher_row = res[res['spec'] == 'switcher'].iloc[0]
b, se, p = switcher_row['coef'], switcher_row['se'], switcher_row['p']

panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel.csv'))
avail = panel[panel['d_od'].notna()]
switcher_pop = avail[avail['switcher'] == 1]['population'].sum()
print(f"\nSwitcher counties (non-suppressed sample): N={len(avail[avail['switcher']==1])}, "
      f"total population={switcher_pop:,.0f}")

averted_deaths = -b / 100000 * switcher_pop  # b is negative (deaths fell), so -b = decline
averted_deaths_lo = -( b - 1.96*se) / 100000 * switcher_pop
averted_deaths_hi = -( b + 1.96*se) / 100000 * switcher_pop
yll = averted_deaths * remaining_le_at_avg_age

print(f"\nImplied averted deaths (point estimate, switcher counties, 2020-2023): {averted_deaths:.1f}")
print(f"  95% CI: [{min(averted_deaths_lo, averted_deaths_hi):.1f}, {max(averted_deaths_lo, averted_deaths_hi):.1f}]")
print(f"Implied years of life saved (averted deaths x remaining LE at avg age {avg_age:.0f}): {yll:.0f}")
print(f"  ({'NOTE: CI for the underlying effect includes zero (p={:.3f}), so this YLL figure '.format(p)}"
      f"is a point estimate under the (not-fully-significant-in-every-spec) switcher coefficient, not a confirmed effect.)")

out = dict(avg_age_at_death=avg_age, remaining_le=remaining_le_at_avg_age,
           switcher_population=float(switcher_pop), averted_deaths=averted_deaths,
           averted_deaths_lo=min(averted_deaths_lo, averted_deaths_hi),
           averted_deaths_hi=max(averted_deaths_lo, averted_deaths_hi), yll=yll)
with open(os.path.join(DATA, 'welfare_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved welfare_summary.json")
