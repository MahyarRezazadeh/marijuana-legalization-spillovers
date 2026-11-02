# ============================================================
# Chunk-reads a NIBRS Arrestee-Level ICPSR extract .dta file
# (one row per arrestee, full incident/property detail attached),
# isolates marijuana (offense 351 + suspected drug type 5) and
# alcohol (offense 905 Drunkenness / 907 Liquor Law Violations,
# excluding 904 DUI -- same exclusion as the SRS-based Part A
# pipeline) arrests in the 32 never-legalizing states, bins by
# arrestee age, and aggregates to county (FIPS5) x age-bin counts.
# Memory-bounded: only ~20 of 363 columns are ever materialized,
# and each chunk is aggregated down before the next is read.
# ============================================================
import sys, time
import numpy as np
import pandas as pd

YEAR = int(sys.argv[1])
DTA_PATH = sys.argv[2]
OUT_CSV = sys.argv[3]
OUT_UNIVERSE_CSV = sys.argv[4]

NEVER_LEGAL = ['AL','AR','DC','DE','FL','GA','HI','IA','ID','IN','KS','KY','LA','ME',
               'MN','MS','NC','ND','NE','NH','OH','OK','PA','SC','SD','TN','TX','UT',
               'VA','WI','WV','WY']

DRUGTYPE_COLS = ['V30121','V30122','V30123','V30161','V30162','V30163','V30201','V30202','V30203']
NEEDED = ['FIPS_STATE','ORI','INCDATE','BH008','BH054','BH055','V6011','V6014'] + DRUGTYPE_COLS

def bin_ages(arr):
    out = np.full(arr.shape, None, dtype=object)
    out[(arr >= 0) & (arr < 21)] = '<21'
    out[(arr >= 21) & (arr < 31)] = '21-30'
    out[(arr >= 31) & (arr < 46)] = '31-45'
    out[(arr >= 46) & (arr <= 99)] = '46+'
    return out

t0 = time.time()
reader = pd.read_stata(DTA_PATH, columns=NEEDED, convert_categoricals=False,
                        chunksize=200_000, iterator=True)

parts = []
universe = set()  # fips5 of every county with an active, single-county, NIBRS-reporting
                   # agency this year, regardless of offense -- the true sampling frame,
                   # so a county absent from `parts` can be distinguished as a real zero
                   # (in universe) vs. simply uncovered by NIBRS that year (not in universe).
n_seen = 0
n_kept = 0
for i, chunk in enumerate(reader):
    n_seen += len(chunk)
    c = chunk[chunk['BH008'].isin(NEVER_LEGAL)]
    # single-county agencies only, matching Part A. "No 2nd county" is coded
    # as NaN in some extract years and as a negative sentinel (e.g. -6) in
    # others, so treat both as "not applicable" -> single-county agency.
    bh055_num = pd.to_numeric(c['BH055'], errors='coerce')
    c = c[bh055_num.isna() | (bh055_num < 0)]
    c = c[pd.to_numeric(c['BH054'], errors='coerce') >= 0]
    yr = c['INCDATE'].str[-4:]
    c = c[yr == str(YEAR)]
    if len(c) == 0:
        continue

    fips5_all = c['FIPS_STATE'].astype(str).str.zfill(2) + c['BH054'].astype(int).astype(str).str.zfill(3)
    universe.update(fips5_all.unique().tolist())

    is_mj = (c['V6011'] == 351) & (c[DRUGTYPE_COLS] == 5).any(axis=1)
    is_alc = c['V6011'].isin([905, 907])
    c = c[is_mj | is_alc].copy()
    if len(c) == 0:
        continue

    c['outcome'] = np.where(is_mj[c.index], 'mj', 'alc')
    c['age_bin'] = bin_ages(c['V6014'].to_numpy())
    c = c[c['age_bin'].notna()]
    c['fips5'] = fips5_all.loc[c.index]

    agg = c.groupby(['fips5', 'age_bin', 'outcome']).size().reset_index(name='n')
    parts.append(agg)
    n_kept += len(c)
    if (i + 1) % 5 == 0:
        print(f"  chunk {i+1}: seen={n_seen:,} kept_mj_alc={n_kept:,} elapsed={time.time()-t0:.0f}s", flush=True)

reader.close()
print(f"DONE. seen={n_seen:,} kept={n_kept:,} counties_in_universe={len(universe):,} elapsed={time.time()-t0:.0f}s")
pd.Series(sorted(universe), name='fips5').to_csv(OUT_UNIVERSE_CSV, index=False)

result = pd.concat(parts, ignore_index=True).groupby(['fips5', 'age_bin', 'outcome'])['n'].sum().reset_index()
result['year'] = YEAR
result = result.pivot_table(index=['fips5', 'year', 'age_bin'], columns='outcome', values='n', fill_value=0).reset_index()
for col in ['mj', 'alc']:
    if col not in result.columns:
        result[col] = 0
result.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} ({len(result)} rows)")
print(result.groupby('age_bin')[['mj','alc']].sum())
