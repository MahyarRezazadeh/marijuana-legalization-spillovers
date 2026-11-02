"""Aggregate FARS accident.csv (2020-2024) into county-year total traffic-fatality counts."""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'fars')

rows = []
for y in range(2020, 2025):
    path = os.path.join(RAW, str(y), f'FARS{y}NationalCSV', 'accident.csv')
    df = pd.read_csv(path, low_memory=False, encoding='latin1')
    cols = [c.strip().upper() for c in df.columns]
    cols[0] = 'STATE'  # first column is always STATE; some years' files have a stray BOM prefix under latin1
    df.columns = cols
    df = df[['STATE', 'COUNTY', 'YEAR', 'FATALS']]
    df['fips5'] = df['STATE'].astype(int).astype(str).str.zfill(2) + df['COUNTY'].astype(int).astype(str).str.zfill(3)
    agg = df.groupby('fips5', as_index=False)['FATALS'].sum().rename(columns={'FATALS': 'fatalities'})
    agg['year'] = y
    rows.append(agg)
    print(f"{y}: {len(df)} crashes, {agg['fatalities'].sum()} total fatalities, {agg['fips5'].nunique()} counties")

out = pd.concat(rows, ignore_index=True)
out.to_csv(os.path.join(BASE, 'data', 'fars_county_year_fatalities.csv'), index=False)
print(f"\nSaved fars_county_year_fatalities.csv: {len(out)} county-year rows")
