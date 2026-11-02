import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_MI = 3958.8
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# never-legal county universe (fips5, state_abbr, population, lat, lon), incl. always-near
counties = pd.read_csv(
    r"..\data\distance_by_year_rec_full.csv",
    usecols=["fips5", "state_abbr", "population", "lat", "lon"],
)
counties["fips5"] = counties["fips5"].astype(str).str.zfill(5)

disp = pd.read_csv(r"dispensary listings\dispensaries_master_geocoded.csv")
disp = disp.dropna(subset=["lat", "lon"])
disp["open_date"] = pd.to_datetime(disp["open_date"], errors="coerce")
disp = disp.dropna(subset=["open_date"])
print(f"Usable geocoded, dated dispensaries: {disp.shape[0]} of 5825")
print(disp["date_confidence"].value_counts())

county_rad = np.radians(counties[["lat", "lon"]].to_numpy())

for year in YEARS:
    cutoff = pd.Timestamp(f"{year}-12-31")
    open_now = disp[disp["open_date"] <= cutoff]
    tree = BallTree(np.radians(open_now[["lat", "lon"]].to_numpy()), metric="haversine")
    dist_rad, _ = tree.query(county_rad, k=1)
    counties[f"dist_{year}"] = dist_rad[:, 0] * EARTH_RADIUS_MI
    print(year, "open dispensaries:", len(open_now), "mean county dist:", counties[f"dist_{year}"].mean().round(1))

counties.to_csv("data/distance_by_year_dispensary.csv", index=False)
print(counties.head())
