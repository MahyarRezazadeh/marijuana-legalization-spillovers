# ============================================================
# Journal of Health Economics draft -- welfare (averted deaths,
# years-of-life-saved) translation, using the switch-year (t=0)
# event-study coefficient from each design. Same methodology
# established earlier in this project (see welfare_summary_
# pretrend_robust.json): YLL-per-averted-death = state-population-
# weighted average life expectancy of switcher-county states MINUS
# the national average age at overdose death (46.14, from the
# existing national 2024 age-distribution pull) -- a literal
# LE-minus-age-at-death figure per the user's explicit instruction
# in this project, not full remaining actuarial life expectancy.
#
# Reuses (does not re-pull): avg_age_at_death.json and
# state_life_expectancy_2021.csv, both already in
# ..\Economic Letters\{tables,data}\.
# ============================================================
import os, json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
EL = os.path.join(BASE, '..', 'Economic Letters')

with open(os.path.join(EL, 'tables', 'avg_age_at_death.json')) as f:
    avg_age = json.load(f)['avg_age_at_death']

life_exp = pd.read_csv(os.path.join(EL, 'data', 'state_life_expectancy_2021.csv'))

STATE_NAME_TO_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY',
}
life_exp['state_abbr'] = life_exp['state'].map(STATE_NAME_TO_ABBR)

def compute(design_name, panel_file, results_file):
    panel = pd.read_csv(os.path.join(DATA, panel_file), dtype={'fips5': str})
    with open(os.path.join(DATA, results_file)) as f:
        res = json.load(f)
    switch0 = panel[(panel['bucket'] == 'et_0') & panel['baseline_rate'].notna()]

    merged = switch0.merge(life_exp[['state_abbr', 'life_expectancy']], on='state_abbr', how='left')
    missing = merged['life_expectancy'].isna().sum()
    weighted_le = (merged['life_expectancy'] * merged['population']).sum() / merged['population'].sum()
    le_diff = weighted_le - avg_age

    coef = res['adjusted']['et_0']['coef']
    se = res['adjusted']['et_0']['se']
    pop = merged['population'].sum()

    averted = abs(coef) * (pop / 1e5)
    averted_lo = abs(coef - 1.96 * se) * (pop / 1e5)
    averted_hi = abs(coef + 1.96 * se) * (pop / 1e5)
    yll = averted * le_diff

    print(f"=== {design_name} ===")
    print(f"  Switch-year (t=0) counties: {merged['fips5'].nunique()} ({missing} missing a state life-expectancy match)")
    print(f"  Population: {pop:,.0f}")
    print(f"  Coefficient: {coef:+.4f} (se={se:.4f})")
    print(f"  State-population-weighted life expectancy: {weighted_le:.2f}; national avg age at death: {avg_age:.2f}; LE_DIFF: {le_diff:.2f}")
    print(f"  Averted deaths: {averted:,.0f} (95% CI [{min(averted_lo,averted_hi):,.0f}, {max(averted_lo,averted_hi):,.0f}])")
    print(f"  Years of life saved: {yll:,.0f}\n")

    return dict(design=design_name, coef=coef, se=se, population=float(pop), weighted_life_expectancy=float(weighted_le),
                avg_age_at_death=float(avg_age), le_diff=float(le_diff), averted_deaths=float(averted),
                averted_deaths_lo=float(min(averted_lo, averted_hi)), averted_deaths_hi=float(max(averted_lo, averted_hi)),
                yll=float(yll))

out = {}
out['main'] = compute('Main design (distance to legal-market county)', 'eventstudy_main_panel.csv', 'eventstudy_main_results.json')
out['dispensary'] = compute('Dispensary design (distance to operating dispensary)', 'eventstudy_dispensary_panel.csv', 'eventstudy_dispensary_results.json')

with open(os.path.join(DATA, 'welfare_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved welfare_summary.json")
