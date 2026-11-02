"""Merge CO+WA county-year revenue with distance-to-nonlegal-border data into one analysis panel."""
import pandas as pd

geo = pd.read_csv("data/county_universe_with_distance.csv")
geo = geo[geo["state_abbr"].isin(["CO", "WA"])].copy()
geo["county_key"] = geo["county_name"].str.replace(" County", "", regex=False).str.strip().str.upper()

co = pd.read_csv("data/co_revenue_annual.csv")
wa = pd.read_csv("data/wa_revenue_annual.csv")
wa = wa.drop(columns=["excise_tax"])

rev = pd.concat([co, wa], ignore_index=True)

panel = geo.merge(rev, on=["state_abbr", "county_key"], how="inner")

unmatched_rev = rev.merge(geo[["state_abbr", "county_key"]], on=["state_abbr", "county_key"], how="left", indicator=True)
unmatched_rev = unmatched_rev[unmatched_rev["_merge"] == "left_only"]
if len(unmatched_rev):
    print("WARNING unmatched revenue county-years (dropped):")
    print(unmatched_rev[["state_abbr", "county_key", "fy_year"]].drop_duplicates().to_string(index=False))

panel["log_dist_nonlegal"] = panel["dist_nonlegal_mi"].apply(lambda x: __import__("numpy").log(x))
panel["border_50mi"] = (panel["dist_nonlegal_mi"] <= 50).astype(int)
panel["revenue_per_capita"] = panel["total_sales"] / panel["population"]
panel["log_revenue_per_capita"] = panel["revenue_per_capita"].apply(lambda x: __import__("numpy").log(x) if x > 0 else float("nan"))

panel.to_csv("data/county_revenue_panel.csv", index=False)
print(f"\nFinal panel: {len(panel)} county-year obs, {panel['fips5'].nunique()} counties, states: {panel['state_abbr'].value_counts().to_dict()}")
print(f"Year range: {panel['fy_year'].min()}-{panel['fy_year'].max()}")
print(panel[["state_abbr", "fy_year", "county_key", "total_sales", "population", "dist_nonlegal_mi", "revenue_per_capita"]].head(10))
