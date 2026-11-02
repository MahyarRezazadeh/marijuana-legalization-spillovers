import pandas as pd
import numpy as np
import json
from sklearn.neighbors import BallTree

EARTH_RADIUS_MI = 3958.8
THRESH = 50.0
YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
TRANSITIONS = list(zip(YEARS[:-1], YEARS[1:]))

# first-wave legal-market states and their retail-opening dates
FIRST_WAVE = {"CO": "2014-01-01", "WA": "2014-07-08", "OR": "2015-10-01",
              "AK": "2016-10-01", "CA": "2018-01-01"}
# states that ALSO legalize inside this 2013-2019 window and would contaminate
# "never-legal" control status if left in (Nevada 2017-07-01, Massachusetts 2018-11-20)
AMBIGUOUS = ["NV", "MA"]
EXCLUDE_FROM_SAMPLE = list(FIRST_WAVE.keys()) + AMBIGUOUS

with open("data/nchs_annual_county_2013_2019.json", encoding="utf-8-sig") as f:
    raw = json.load(f)
nchs = pd.DataFrame(raw)
nchs["fips5"] = nchs["fips"].astype(str).str.zfill(5)
nchs["year"] = nchs["year"].astype(int)
nchs["model_based_death_rate"] = pd.to_numeric(nchs["model_based_death_rate"], errors="coerce")
nchs["population"] = pd.to_numeric(nchs["population"], errors="coerce")

centroids = pd.read_stata(r"..\..\..\data\county_centroids.dta")
centroids["fips5"] = centroids["fips5"].astype(str).str.zfill(5)

counties = centroids.merge(
    nchs[nchs["year"] == 2013][["fips5", "state"]].rename(columns={"state": "state_full"}),
    on="fips5", how="inner"
)

legal_source = centroids[centroids["state_abbr"].isin(FIRST_WAVE.keys())]

# which first-wave states are open by Dec 31 of each year
def open_states(year):
    cutoff = pd.Timestamp(f"{year}-12-31")
    return [s for s, d in FIRST_WAVE.items() if pd.Timestamp(d) <= cutoff]

sample = centroids[~centroids["state_abbr"].isin(EXCLUDE_FROM_SAMPLE)].copy()
print(f"Never-legal sample universe (excl. first-wave + NV/MA): {sample.shape[0]} counties")

sample_rad = np.radians(sample[["lat", "lon"]].to_numpy())
for year in YEARS:
    open_now = legal_source[legal_source["state_abbr"].isin(open_states(year))]
    if len(open_now) == 0:
        sample[f"dist_{year}"] = 9999.0
        print(year, "legal states open:", open_states(year), "(none yet)")
        continue
    tree = BallTree(np.radians(open_now[["lat", "lon"]].to_numpy()), metric="haversine")
    dist_rad, _ = tree.query(sample_rad, k=1)
    sample[f"dist_{year}"] = dist_rad[:, 0] * EARTH_RADIUS_MI
    print(year, "legal states open:", open_states(year), "mean dist:", round(sample[f"dist_{year}"].mean(), 1))

sample.to_csv("data/distance_by_year_firstwave.csv", index=False)

# always-near at 2013 baseline: should be ~0 since no state is legal yet in 2013
always_near = sample[sample["dist_2013"] < THRESH]["fips5"]
base = sample[~sample["fips5"].isin(always_near)].copy()
print(f"Dropped {len(always_near)} always-near-at-2013-baseline counties (expect 0); {base.shape[0]} remain")

od = nchs.set_index(["fips5", "year"])["model_based_death_rate"]

rows = []
switched = set()
for year_pre, year_post in TRANSITIONS:
    label = f"{year_pre}-{year_post}"
    active = base[~base["fips5"].isin(switched)]
    dpre, dpost = active[f"dist_{year_pre}"], active[f"dist_{year_post}"]
    is_switcher = (dpre >= THRESH) & (dpost < THRESH)
    rate_pre = active["fips5"].map(lambda f: od.get((f, year_pre), np.nan))
    rate_post = active["fips5"].map(lambda f: od.get((f, year_post), np.nan))
    chunk = pd.DataFrame({
        "fips5": active["fips5"], "state_abbr": active["state_abbr"],
        "transition": label, "year_pre": year_pre, "year_post": year_post,
        "dist_pre": dpre.values, "dist_post": dpost.values,
        "switcher": is_switcher.astype(int).values,
        "d_dist_mi": (dpost - dpre).values,
        "rate_pre": rate_pre.values, "rate_post": rate_post.values,
    })
    rows.append(chunk)
    switched.update(active.loc[is_switcher, "fips5"])

panel = pd.concat(rows, ignore_index=True)
panel["d_rate"] = panel["rate_post"] - panel["rate_pre"]
baseline_2013 = od.xs(2013, level="year")
panel["baseline_rate"] = panel["fips5"].map(baseline_2013)

print(f"\nStacked panel rows: {panel.shape[0]}, switch events: {panel['switcher'].sum()}")
print(panel.groupby("transition").agg(n=("switcher", "size"), switches=("switcher", "sum"),
                                       mean_ddist=("d_dist_mi", "mean")))

panel.to_csv("data/stacked_panel_firstwave.csv", index=False)
