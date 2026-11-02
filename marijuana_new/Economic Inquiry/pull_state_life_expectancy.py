# ============================================================
# Pulls CDC's "U.S. State Life Expectancy by Sex, 2021" (Socrata
# id it4f-frdc, data.cdc.gov -- not blocked, unlike cdc.gov/wonder)
# for the welfare-translation rewrite: years-of-life-saved should
# use (life expectancy - average age at overdose death), not the
# full remaining actuarial life expectancy, per user instruction.
# State life expectancy is used (population-weighted across
# switcher-county states); average age at overdose death stays
# national since no state-level age-at-death breakdown for
# overdose deaths specifically is publicly available (confirmed in
# earlier sessions).
# ============================================================
import os, json, urllib.request
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

def q(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

rows = q("https://data.cdc.gov/resource/it4f-frdc.json?sex=Total&$limit=200")
df = pd.DataFrame(rows)[['area', 'leb', 'se']]
df.columns = ['state', 'life_expectancy', 'se']
df['life_expectancy'] = pd.to_numeric(df['life_expectancy'], errors='coerce')
df = df[~df['state'].isin(['United States'])].dropna(subset=['life_expectancy'])
df.to_csv(os.path.join(DATA, 'state_life_expectancy_2021.csv'), index=False)
print(f"Saved state_life_expectancy_2021.csv ({len(df)} states)")
print(df.sort_values('life_expectancy').head(3))
print(df.sort_values('life_expectancy').tail(3))
