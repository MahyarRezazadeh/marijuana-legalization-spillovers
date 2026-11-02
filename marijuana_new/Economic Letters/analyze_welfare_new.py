import json
import pandas as pd

with open('data/overdose_results_new.json') as f:
    res = json.load(f)
b, se, p, n = res['switcher']

main = pd.read_csv('data/overdose_county_panel_new.csv')
avail = main[main['d_od'].notna()]
sw_pop = avail[avail['switcher'] == 1]['population'].sum()

remaining_le = 33.4  # same approximation as before (CDC/SSA primary tables unreachable)
averted = -b / 100000 * sw_pop
averted_lo = -(b - 1.96*se) / 100000 * sw_pop
averted_hi = -(b + 1.96*se) / 100000 * sw_pop
yll = averted * remaining_le

print(f"Switcher population: {sw_pop:,.0f}")
print(f"Averted deaths: {averted:.1f}  95% CI [{min(averted_lo,averted_hi):.1f}, {max(averted_lo,averted_hi):.1f}]")
print(f"Years of life saved: {yll:.0f}")

out = dict(switcher_population=float(sw_pop), averted_deaths=averted,
           averted_lo=min(averted_lo, averted_hi), averted_hi=max(averted_lo, averted_hi), yll=yll)
with open('data/welfare_summary_new.json', 'w') as f:
    json.dump(out, f, indent=2)
