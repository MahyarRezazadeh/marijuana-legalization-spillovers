# ============================================================
# National age distribution of drug-overdose deaths (CDC DQS,
# most recent year = 2024), used to (a) show which age groups
# drive overdose mortality nationally as descriptive context,
# and (b) compute the average age at overdose death and implied
# expected years of life lost (YLL) per death, for a welfare-
# relevant translation of the county-level mortality effect.
# Not usable in the county-level regression design itself --
# no county- or state-level age breakdown exists publicly.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

# ── Pull national age-specific overdose death rates (2024, "All drug overdose deaths") ──
url = "https://data.cdc.gov/resource/rdjz-vn2n.json?%24limit=5000"
with urllib.request.urlopen(url, timeout=30) as resp:
    raw = json.loads(resp.read().decode())
df = pd.DataFrame(raw)
sub = df[(df['group'] == 'Age group') & (df['time_period'] == '2024') &
         (df['subtopic'] == 'All drug overdose deaths')].copy()
sub['estimate'] = pd.to_numeric(sub['estimate'], errors='coerce')
sub = sub[['subgroup', 'estimate']].drop_duplicates('subgroup')
print("2024 national overdose death rate by age group (crude, per 100,000):")
print(sub.to_string(index=False))
sub.to_csv(os.path.join(DATA, 'national_age_overdose_rates_2024.csv'), index=False)

# ── Age-group midpoints (open-ended top bin: use 90 as a conventional cap) ──
AGE_MIDPOINT = {
    'Younger than 15 years': 7.5, '15-24 years': 19.5, '25-34 years': 29.5,
    '35-44 years': 39.5, '45-54 years': 49.5, '55-64 years': 59.5,
    '65-74 years': 69.5, '75-84 years': 79.5, '85 years and older': 90.0,
}
sub['midpoint'] = sub['subgroup'].map(AGE_MIDPOINT)

# ── Population by single year of age (2023 vintage, closest available to 2024 rates) ──
pop_url = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/national/asrh/nc-est2023-agesex-res.csv"
pop = pd.read_csv(pop_url, encoding='latin1')
pop = pop[(pop['SEX'] == 0) & (pop['AGE'] != 999)][['AGE', 'POPESTIMATE2023']].rename(columns={'POPESTIMATE2023': 'pop'})

BIN_RANGES = {
    'Younger than 15 years': (0, 14), '15-24 years': (15, 24), '25-34 years': (25, 34),
    '35-44 years': (35, 44), '45-54 years': (45, 54), '55-64 years': (55, 64),
    '65-74 years': (65, 74), '75-84 years': (75, 84), '85 years and older': (85, 100),
}

rows = []
for _, r in sub.iterrows():
    lo, hi = BIN_RANGES[r['subgroup']]
    bin_pop = pop[(pop['AGE'] >= lo) & (pop['AGE'] <= hi)]
    total_pop = bin_pop['pop'].sum()
    avg_age_in_bin = (bin_pop['AGE'] * bin_pop['pop']).sum() / total_pop
    implied_deaths = r['estimate'] / 100000 * total_pop
    rows.append(dict(subgroup=r['subgroup'], rate=r['estimate'], population=total_pop,
                      avg_age_in_bin=avg_age_in_bin, implied_deaths=implied_deaths))

age_dist = pd.DataFrame(rows)
age_dist['share_of_deaths'] = age_dist['implied_deaths'] / age_dist['implied_deaths'].sum()
print("\n" + age_dist.to_string(index=False))

avg_age_at_death = (age_dist['implied_deaths'] * age_dist['avg_age_in_bin']).sum() / age_dist['implied_deaths'].sum()
print(f"\nImplied total US overdose deaths, 2024 (from age-specific rates x population): "
      f"{age_dist['implied_deaths'].sum():,.0f}")
print(f"Population-weighted average age at overdose death: {avg_age_at_death:.1f} years")

age_dist.to_csv(os.path.join(DATA, 'national_age_distribution_overdose_2024.csv'), index=False)
with open(os.path.join(DATA, 'avg_age_at_death.json'), 'w') as f:
    json.dump({'avg_age_at_death': avg_age_at_death,
               'total_implied_deaths_2024': float(age_dist['implied_deaths'].sum())}, f)
print("\nSaved national_age_distribution_overdose_2024.csv, avg_age_at_death.json")
