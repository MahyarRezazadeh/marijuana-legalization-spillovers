import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_MI = 3958.8
THRESH = 50.0

centroids = pd.read_stata(r"..\..\..\data\county_centroids.dta")
centroids["fips5"] = centroids["fips5"].astype(str).str.zfill(5)

norc = pd.read_excel("data/norc_overdose_raw.xlsx", sheet_name="data")
norc = norc[norc["GEOID"].apply(lambda x: isinstance(x, int) and x >= 1000)].copy()
norc["fips5"] = norc["GEOID"].astype(int).astype(str).str.zfill(5)

for col in ["Opioid Death Count (2010-2013)", "Opioid Death Count (2014-2017)",
            "Crude Opioid Death Rate (2010-2013)", "Crude Opioid Death Rate (2014-2017)",
            "ACS Total Population (2009-2013)", "ACS Total Population (2013-2017)"]:
    norc[col] = pd.to_numeric(norc[col], errors="coerce")

df = centroids.merge(norc, on="fips5", how="inner")
df["state_fips2"] = df["fips5"].str[:2]

CO_WA_FIPS = df.loc[df["state_abbr"].isin(["CO", "WA"]), "state_fips2"].unique()
AMBIGUOUS = ["CO", "WA", "OR", "AK"]  # legalized at/inside the 2014-2017 window itself

legal_source = df[df["state_abbr"].isin(["CO", "WA"])]
tree = BallTree(np.radians(legal_source[["lat", "lon"]].to_numpy()), metric="haversine")

sample = df[~df["state_abbr"].isin(AMBIGUOUS)].copy()
dist_rad, _ = tree.query(np.radians(sample[["lat", "lon"]].to_numpy()), k=1)
sample["dist_mi"] = dist_rad[:, 0] * EARTH_RADIUS_MI
sample["switcher"] = (sample["dist_mi"] < THRESH).astype(int)

sample["rate_pre"] = sample["Crude Opioid Death Rate (2010-2013)"]
sample["rate_post"] = sample["Crude Opioid Death Rate (2014-2017)"]
sample["d_rate"] = sample["rate_post"] - sample["rate_pre"]

usable = sample.dropna(subset=["rate_pre", "rate_post"])
print(f"Never-legal-in-either-period universe (excl. CO/WA/OR/AK): {sample.shape[0]} counties")
print(f"Usable (non-suppressed both periods): {usable.shape[0]} counties")
print(f"Switchers (<{THRESH}mi of CO/WA): {usable['switcher'].sum()}, controls: {(usable['switcher']==0).sum()}")
print()
print("Raw group means (Δ opioid death rate per 100k):")
print(usable.groupby("switcher")["d_rate"].agg(["mean", "count"]))

usable.to_csv("data/firstwave_cowa_panel.csv", index=False)
