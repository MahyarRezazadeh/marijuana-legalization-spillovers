import pandas as pd

cols = ["uid", "input_address", "match_status", "match_type", "matched_address", "coords", "tigerline_id", "side"]
geo = pd.read_csv("geocode_output.csv", header=None, names=cols, dtype={"uid": int})
geo[["lon", "lat"]] = geo["coords"].str.split(",", expand=True).astype(float)

crosswalk = pd.read_csv("geocode_crosswalk.csv")
merged = crosswalk.merge(geo[["uid", "match_status", "match_type", "lat", "lon"]], on="uid", how="left")

print("Overall match rate:", (merged["match_status"] == "Match").mean())
print(merged.groupby("state_src")["match_status"].value_counts())

merged.to_csv("geocode_results.csv", index=False)
