"""Stream-filter NASS bulk Census-of-Agriculture files for county-level farmland asset value per acre.
Chunked read (2.2GB RAM machine) -- never loads the full ~300MB-per-year file at once.
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'nass')

TARGET_ITEM = 'AG LAND, INCL BUILDINGS - ASSET VALUE, MEASURED IN $ / ACRE'

FILES = {
    2022: 'qs.census2022.txt.gz',
    2017: 'qs.census2017.txt.gz',
    2012: 'qs.census2012.txt.gz',
}

frames = []
for year, fname in FILES.items():
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        print(f"SKIP {year}: {fname} not downloaded")
        continue
    matched_chunks = []
    for chunk in pd.read_csv(path, sep='\t', encoding='latin1', chunksize=200_000,
                              usecols=['SHORT_DESC', 'AGG_LEVEL_DESC', 'STATE_ALPHA', 'STATE_NAME', 'STATE_FIPS_CODE',
                                       'COUNTY_NAME', 'COUNTY_ANSI', 'YEAR', 'VALUE'],
                              dtype=str, low_memory=False):
        hit = chunk[(chunk['SHORT_DESC'] == TARGET_ITEM) & (chunk['AGG_LEVEL_DESC'] == 'COUNTY')]
        if len(hit):
            matched_chunks.append(hit)
    if matched_chunks:
        out = pd.concat(matched_chunks, ignore_index=True)
        frames.append(out)
        print(f"{year}: {len(out)} county rows matched")
    else:
        print(f"{year}: 0 rows matched")

result = pd.concat(frames, ignore_index=True)
result.to_csv(os.path.join(BASE, 'data', 'nass_land_value_per_acre_county.csv'), index=False)
print(f"\nSaved nass_land_value_per_acre_county.csv: {len(result)} total rows")
print(result['VALUE'].value_counts().head(10))
