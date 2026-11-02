# ============================================================
# Descriptive table: how many counties (and how much population)
# would be classified "exposed" under alternative distance
# thresholds, December 2023 snapshot. Same data/exclusions as the
# treatment/control map (Hawaii excluded -- distance not
# comparable) and the discontinuity-check figures. Purely
# descriptive -- no regression, just counts -- meant to visualize
# the sample-size-vs-misclassification tradeoff behind the paper's
# 50-mile choice.
# ============================================================
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
EL = os.path.dirname(BASE)
TABLES = os.path.join(BASE, 'tables')

dist = pd.read_csv(os.path.join(EL, 'data', 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
d = dist[dist['state_abbr'] != 'HI'].copy()

THRESHOLDS = [20, 25, 40, 50, 70, 100]
total_pop = d['population'].sum()
total_n = len(d)

rows = []
for t in THRESHOLDS:
    exposed = d[d['dist_2023'] < t]
    control = d[d['dist_2023'] >= t]
    n_exp = len(exposed)
    n_ctrl = len(control)
    pop_exp = exposed['population'].sum()
    rows.append(dict(
        threshold=t,
        n_exposed=n_exp,
        n_control=n_ctrl,
        pct_exposed_counties=100 * n_exp / total_n,
        pop_exposed=int(pop_exp),
        pct_exposed_pop=100 * pop_exp / total_pop,
    ))

res = pd.DataFrame(rows)
pd.set_option('display.float_format', lambda x: f'{x:.1f}')
print(f"Total counties (never-legal universe, ex-HI): {total_n}, total population: {total_pop:,.0f}")
print(res.to_string(index=False))

res.to_csv(os.path.join(TABLES, 'threshold_composition.csv'), index=False)
print("\nSaved tables/threshold_composition.csv")
