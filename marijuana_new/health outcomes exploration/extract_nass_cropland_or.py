"""Stream-filter NASS bulk Census-of-Agriculture files for Oregon county-level harvested cropland acres."""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, 'data', 'raw', 'nass')

TARGET_ITEM = 'AG LAND, CROPLAND, HARVESTED - ACRES'

FILES = {2022: 'qs.census2022.txt.gz', 2017: 'qs.census2017.txt.gz', 2012: 'qs.census2012.txt.gz'}

frames = []
for year, fname in FILES.items():
    path = os.path.join(RAW, fname)
    matched_chunks = []
    for chunk in pd.read_csv(path, sep='\t', encoding='latin1', chunksize=200_000,
                              usecols=['SHORT_DESC', 'AGG_LEVEL_DESC', 'STATE_ALPHA', 'STATE_FIPS_CODE',
                                       'DOMAIN_DESC', 'COUNTY_NAME', 'COUNTY_ANSI', 'YEAR', 'VALUE'],
                              dtype=str, low_memory=False):
        hit = chunk[(chunk['SHORT_DESC'] == TARGET_ITEM) & (chunk['AGG_LEVEL_DESC'] == 'COUNTY')
                    & (chunk['STATE_ALPHA'] == 'OR') & (chunk['DOMAIN_DESC'] == 'TOTAL')]
        if len(hit):
            matched_chunks.append(hit)
    if matched_chunks:
        out = pd.concat(matched_chunks, ignore_index=True)
        frames.append(out)
        print(f"{year}: {len(out)} OR county rows matched")
    else:
        print(f"{year}: 0 rows matched")

result = pd.concat(frames, ignore_index=True)
result.to_csv(os.path.join(BASE, 'data', 'nass_cropland_harvested_or_county.csv'), index=False)
print(f"\nSaved: {len(result)} total rows")
print(result[['COUNTY_NAME', 'YEAR', 'VALUE']].head(15))
