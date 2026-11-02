import json
import pandas as pd

with open('data/overdose_results_new.json') as f:
    res = json.load(f)
with open('data/welfare_summary_new.json') as f:
    wel = json.load(f)

rows = []
for spec in ['switcher', 'dist_mile', 'log_dist']:
    b, se, p, n = res[spec]
    rows.append(dict(spec=spec, coef=b, se=se, p=p, n=n))
pd.DataFrame(rows).to_csv('tables/tE1_overdose_regressions_new.csv', index=False)

summary = pd.DataFrame([
    {'Variable': 'Switcher counties, full panel (N)', 'Value': 287},
    {'Variable': 'Control counties, full panel (N)', 'Value': 1928},
    {'Variable': 'Always-near counties, full panel (N)', 'Value': 116},
    {'Variable': 'Switcher counties, non-suppressed (N)', 'Value': 98},
    {'Variable': 'Control counties, non-suppressed (N)', 'Value': 798},
    {'Variable': 'Switcher population (non-suppressed)', 'Value': f"{wel['switcher_population']:,.0f}"},
    {'Variable': 'Implied averted deaths', 'Value': f"{wel['averted_deaths']:.0f}"},
    {'Variable': 'Implied averted deaths, 95% CI', 'Value': f"[{wel['averted_lo']:.0f}, {wel['averted_hi']:.0f}]"},
    {'Variable': 'Implied years of life saved', 'Value': f"{wel['yll']:.0f}"},
])
summary.to_csv('tables/tE0_overdose_summary_new.csv', index=False)
print(rows)
print(summary)
