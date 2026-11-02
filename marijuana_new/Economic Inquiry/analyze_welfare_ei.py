# ============================================================
# Welfare translation, rewritten per user instruction: years of
# life saved = (life expectancy - average age at overdose death)
# per averted death, NOT the full remaining actuarial life
# expectancy at the average age of death (which is what the
# Economic Letters version used). Life expectancy is state-level
# (population-weighted across switcher-county states); average age
# at overdose death stays national (no state-level breakdown of
# overdose deaths by age is publicly available).
# ============================================================
import os, json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MARIJUANA_NEW = os.path.dirname(BASE)
EL = os.path.join(MARIJUANA_NEW, 'Economic Letters')
DATA = os.path.join(BASE, 'data')

with open(os.path.join(EL, 'data', 'multiperiod_results_rec.json')) as f:
    res = json.load(f)
b, se = res['switcher']['coef'], res['switcher']['se']

s = pd.read_csv(os.path.join(EL, 'data', 'stacked_panel_rec_full.csv'), dtype={'fips5': str})
avail = s[s['d_od'].notna() & (s['population'] > 0)]
sw = avail[avail['switcher'] == 1].drop_duplicates('fips5')
pop = sw['population'].sum()
n_counties = len(sw)

le = pd.read_csv(os.path.join(DATA, 'state_life_expectancy_2021.csv'))
STATE_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC',
}
le['state_abbr'] = le['state'].map(STATE_ABBR)

sw = sw.merge(le[['state_abbr', 'life_expectancy']], on='state_abbr', how='left')
missing = sw[sw['life_expectancy'].isna()]
if len(missing):
    print(f"WARNING: {len(missing)} switcher counties with no state LE match: {missing['state_abbr'].unique()}")
sw = sw.dropna(subset=['life_expectancy'])
weighted_le = (sw['life_expectancy'] * sw['population']).sum() / sw['population'].sum()

AVG_AGE_AT_DEATH = 46.1  # national, from CDC age-specific overdose rates (unchanged data source)
LE_DIFF = weighted_le - AVG_AGE_AT_DEATH

def averted(coef):
    return -coef * pop / 1e5

pt = averted(b)
lo = averted(b + 1.96 * se)
hi = averted(b - 1.96 * se)
yll = pt * LE_DIFF

print(f"Switcher counties: {n_counties}, total population: {pop:,.0f}")
print(f"Population-weighted state life expectancy (switcher counties): {weighted_le:.2f} years")
print(f"National average age at overdose death: {AVG_AGE_AT_DEATH} years")
print(f"Life-expectancy gap (years of life saved per averted death): {LE_DIFF:.2f}")
print(f"Averted deaths (point): {pt:.1f}  [95% CI: {lo:.1f}, {hi:.1f}]")
print(f"Years of life saved: {yll:,.0f}  [95% CI: {lo*LE_DIFF:,.0f}, {hi*LE_DIFF:,.0f}]")

out = dict(n_switcher_counties=n_counties, switcher_population=int(pop),
           coef=b, se=se, averted_deaths=pt, averted_lo=lo, averted_hi=hi,
           weighted_state_le=weighted_le, avg_age_at_death=AVG_AGE_AT_DEATH,
           le_diff=LE_DIFF, yll=yll, yll_lo=lo*LE_DIFF, yll_hi=hi*LE_DIFF)
with open(os.path.join(DATA, 'welfare_summary_ei.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved welfare_summary_ei.json")
