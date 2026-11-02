import sys
import pandas as pd
import numpy as np

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0

dist = pd.read_csv(r"..\data\distance_by_year_rec_full.csv", dtype={"fips5": str})
dist["fips5"] = dist["fips5"].str.zfill(5)

always_near = dist[dist["dist_2020"] < THRESH]["fips5"]
sample = dist[~dist["fips5"].isin(always_near)].copy()
print(f"Dropped {len(always_near)} always-near-at-2020 counties (same rule as main design); {sample.shape[0]} remain")

# first year each county's state-legal-county distance crosses below 50mi (absorbing
# treatment group, standard staggered-adoption setup); 0 = never treated by 2025
def first_switch_year(row):
    for y in YEARS[1:]:
        if row[f"dist_{y}"] < THRESH:
            return y
    return 0

sample["g"] = sample.apply(first_switch_year, axis=1)
print(sample["g"].value_counts().sort_index())

# overdose outcomes, reused from the main design's panel (identical CDC VSRR pull)
outcomes = pd.read_csv(r"..\data\stacked_panel_rec_full.csv", dtype={"fips5": str})
outcomes["fips5"] = outcomes["fips5"].str.zfill(5)
od = outcomes.drop_duplicates(subset="fips5").set_index("fips5")[[f"od_{y}" for y in YEARS]]

long_rows = []
for _, r in sample.iterrows():
    fips = r["fips5"]
    if fips not in od.index:
        continue
    od_2020 = od.loc[fips, "od_2020"]
    baseline_rate = od_2020 / r["population"] * 1e5
    for y in YEARS:
        od_count = od.loc[fips, f"od_{y}"]
        long_rows.append({
            "fips5": fips,
            "fips_int": int(fips),
            "state_abbr": r["state_abbr"],
            "year": y,
            "population": r["population"],
            "g": r["g"],
            "od_count": od_count,
            "baseline_rate": baseline_rate,
        })

panel = pd.DataFrame(long_rows)
panel["od_rate"] = panel["od_count"] / panel["population"] * 1e5
panel["treated"] = ((panel["g"] != 0) & (panel["year"] >= panel["g"])).astype(int)

# CS/TWFE both need a balanced-ish panel with no missing outcome across the unit's years;
# drop counties with any suppressed year (same conservative approach as reporting "usable N")
complete = panel.groupby("fips5")["od_rate"].apply(lambda s: s.notna().all())
usable_fips = complete[complete].index
panel_bal = panel[panel["fips5"].isin(usable_fips)].copy()
print(f"Counties with complete 2020-2025 overdose data: {len(usable_fips)} of {panel['fips5'].nunique()}")
print(f"Balanced panel rows: {panel_bal.shape[0]}")

outpath = f"data/panel_twfe_cs_t{int(THRESH)}.csv"
panel_bal.to_csv(outpath, index=False)
print(f"Saved {outpath}")
print(panel_bal.groupby("g")["fips5"].nunique())
