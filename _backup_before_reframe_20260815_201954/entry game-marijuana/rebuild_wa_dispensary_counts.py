# ============================================================
# Corrects a severe undercount discovered in the project's WA
# dispensary data: the master geocoded file has only 98 WA records,
# but WSLCB's own current cannabis retailer list shows 480 active
# retail licenses statewide (462 regular + 2 active title certificate
# + 16 social equity). The old master file's WA raw source
# (wa_raw.csv, 384 total license records across ALL types) was a
# small, stale partial pull, not a processing bug.
#
# This script rebuilds WA county-level ACTIVE dispensary counts
# directly from WSLCB's official retailer list (downloaded 2026-08-11
# from lcb.wa.gov/sites/default/files/2026-08/CannabisApplicants08042026.xlsx,
# "Retailers 8-4-2026" and "SE Retailers 8-4-2026" sheets), which
# already assigns each licensee to a county -- no geocoding needed,
# eliminating that source of error entirely for Washington.
#
# Like the rest of the project's revenue-per-store construction, this
# is a CURRENT (2026-08-04) snapshot applied uniformly across the
# panel's years, the same convention already used for Oregon and
# Montana's cross-sectional per-store checks.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
XLSX = r"C:\Users\mahya\AppData\Local\Temp\claude\C--Users-mahya\afec4191-72bf-46b8-8696-8af88a58f47a\scratchpad\CannabisApplicants.xlsx"

xl = pd.ExcelFile(XLSX)
r1 = xl.parse('Retailers 8-4-2026')
r2 = xl.parse('SE Retailers 8-4-2026')
r2 = r2.rename(columns={'County': 'county', 'Suite/Unit': 'Suite Rm'})

active_status = ['ACTIVE (ISSUED)', 'ACTIVE TITLE CERTIFICATE']
r1_active = r1[r1['Privilege Status'].isin(active_status)].copy()
r2_active = r2[r2['Privilege Status'].isin(active_status)].copy()

print(f"Active regular retailers: {len(r1_active)}")
print(f"Active social-equity retailers: {len(r2_active)}")
print(f"Total active WA retail licenses: {len(r1_active) + len(r2_active)}")

both = pd.concat([r1_active[['Tradename', 'License ', 'county', 'City']],
                   r2_active[['Tradename', 'License ', 'county', 'City']]], ignore_index=True)
both['county_key'] = both['county'].str.strip().str.upper()

counts = both.groupby('county_key').size().rename('dispensary_count_new').reset_index()
print(f"\nWA counties with an active retailer: {len(counts)}")
print(counts.sort_values('dispensary_count_new', ascending=False).to_string(index=False))

counts.to_csv(os.path.join(TABLES, 'wa_dispensary_counts_corrected.csv'), index=False)
print(f"\nSaved wa_dispensary_counts_corrected.csv")

# ---------- compare to the old (undercounted) figures ----------
old = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel.csv'), dtype={'fips5': str})
old_wa = old[(old.state_abbr == 'WA') & (old.fy_year == 2025)][['county_key', 'dispensary_count']]
old_wa = old_wa.rename(columns={'dispensary_count': 'dispensary_count_old'})
compare = counts.merge(old_wa, on='county_key', how='outer').fillna(0)
compare['old_pct_of_true'] = (compare['dispensary_count_old'] / compare['dispensary_count_new'] * 100).round(1)
print("\nOld vs. new dispensary counts by county (old data's coverage rate):")
print(compare.sort_values('dispensary_count_new', ascending=False).to_string(index=False))
