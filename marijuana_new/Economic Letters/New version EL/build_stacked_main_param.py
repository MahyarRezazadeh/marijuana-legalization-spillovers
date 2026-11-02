import sys
import pandas as pd
import numpy as np
import json

THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TRANSITIONS = list(zip(YEARS[:-1], YEARS[1:]))

dist = pd.read_csv(r"..\data\distance_by_year_rec_full.csv", dtype={"fips5": str})
dist["fips5"] = dist["fips5"].str.zfill(5)

always_near = dist[dist["dist_2020"] < THRESH]["fips5"]
sample = dist[~dist["fips5"].isin(always_near)].copy()
print(f"[T={THRESH}] Dropped {len(always_near)} always-near-at-2020 counties; {sample.shape[0]} remain")

with open(r"..\data\overdose_county_multiyear_raw.json", encoding="utf-8") as f:
    raw = json.load(f)
raw_df = pd.DataFrame(raw)
raw_df["fips5"] = raw_df["fips"].astype(str).str.zfill(5)
raw_df["provisional_drug_overdose"] = pd.to_numeric(raw_df["provisional_drug_overdose"], errors="coerce")
od_wide = raw_df.pivot_table(index="fips5", columns="year", values="provisional_drug_overdose", aggfunc="first")
od_wide.columns = [f"od_{int(c)}" for c in od_wide.columns]

rows = []
switched = set()
for year_pre, year_post in TRANSITIONS:
    label = f"{year_pre}-{year_post}"
    active = sample[~sample["fips5"].isin(switched)]
    dpre, dpost = active[f"dist_{year_pre}"], active[f"dist_{year_post}"]
    is_switcher = (dpre >= THRESH) & (dpost < THRESH)
    chunk = pd.DataFrame({
        "fips5": active["fips5"], "state_abbr": active["state_abbr"], "population": active["population"],
        "transition": label, "year_pre": year_pre, "year_post": year_post,
        "switcher": is_switcher.astype(int).values, "d_dist_mi": (dpost - dpre).values,
    })
    rows.append(chunk)
    switched.update(active.loc[is_switcher, "fips5"])

panel = pd.concat(rows, ignore_index=True)
panel = panel.merge(od_wide, on="fips5", how="left")
panel["od_pre"] = panel.apply(lambda r: r[f"od_{r['year_pre']}"], axis=1)
panel["od_post"] = panel.apply(lambda r: r[f"od_{r['year_post']}"], axis=1)
panel["od_rate_pre"] = panel["od_pre"] / panel["population"] * 1e5
panel["od_rate_post"] = panel["od_post"] / panel["population"] * 1e5
panel["d_od"] = panel["od_rate_post"] - panel["od_rate_pre"]
panel["baseline_rate"] = panel["od_2020"] / panel["population"] * 1e5

n_before = panel.shape[0]
panel = panel.dropna(subset=["od_rate_pre", "od_rate_post"])
print(f"[T={THRESH}] Dropped {n_before - panel.shape[0]} suppressed rows; {panel.shape[0]} remain, "
      f"{panel['switcher'].sum()} switch events")

panel.to_csv(f"data/stacked_panel_main_t{int(THRESH)}.csv", index=False)
