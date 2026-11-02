import pandas as pd
import numpy as np

files = {
    "OH": ("ohio_dispensaries.dta", "issuance_date"),
    "NY": ("ny_dispensaries.dta", "retail_date_opened_to_public"),
    "MA": ("ma_dispensaries.dta", "commence_operations_date"),
    "OR": ("or_dispensaries.dta", "effective_date"),
    "WA": ("wa_dispensaries.dta", None),
    "IL": ("il_dispensaries.dta", "issuance_date"),
    "ME": ("me_dispensaries.dta", "issuance_date"),
    "CO": ("co_dispensaries.dta", None),
    "CA": ("ca_dispensaries.dta", "issuance_date"),
}
DATE_CONF = {
    "OH": "high", "NY": "high", "MA": "high", "OR": "high",
    "IL": "high", "ME": "high", "CA": "high",
    "WA": "baseline_assumed", "CO": "baseline_assumed",
}
# WA and CO legalized recreational retail in 2014 -- both markets were already mature
# well before our 2020-2025 panel window starts, so unlike the other 7 states these
# dispensaries contribute no *new-opening* timing variation within the panel. Their
# own license files only carry a renewal/expiration date (not a real open date), and
# using that as a proxy for "when it opened" would be actively misleading (a renewal
# date can postdate the true opening by years with no consistent bias). Instead we
# assign them a fixed pre-panel date so they simply count as "already open" for every
# panel year, matching how the main (county-centroid) design already treats CO/WA.
PRE_PANEL_DATE = pd.Timestamp("2015-01-01")

master = []
for state, (fname, datecol) in files.items():
    df = pd.read_stata(fname)
    df["state"] = state
    if datecol is None:
        df["open_date"] = PRE_PANEL_DATE
    else:
        df["open_date"] = pd.to_datetime(df[datecol], errors="coerce")
    df["date_confidence"] = DATE_CONF[state]
    zip_col = df["zip"] if "zip" in df.columns else df["address"]
    df["zip5"] = zip_col.astype(str).str.extract(r"(\d{5})")[0]
    keep = ["state", "license_number", "address", "zip5", "open_date", "date_confidence"]
    if "latitude" in df.columns:
        df["lat_src"] = df["latitude"]
        df["lon_src"] = df["longitude"]
    else:
        df["lat_src"] = np.nan
        df["lon_src"] = np.nan
    keep += ["lat_src", "lon_src"]
    master.append(df[keep])

master = pd.concat(master, ignore_index=True)
master["uid"] = range(1, len(master) + 1)

# attach Census batch-geocoder results (street-level) where available
geo = pd.read_csv("geocode_results.csv")
geo_match = geo[geo["match_status"] == "Match"][["state_src", "row_idx", "lat", "lon"]]
# row_idx was the original per-state dataframe index at geocode-input time; rebuild the same per-state order
idx_map = []
counters = {s: 0 for s in files}
for _, r in master.iterrows():
    pass  # not used; join instead via state+sequential position matches build_geocode_input.py's construction

# Simpler: geocode_crosswalk.csv uid already lines up 1:1 in the SAME state order as master was built
# (both iterate the same files dict/state list and same per-state row order), so join on state + running count
master["_seq"] = master.groupby("state").cumcount()
geo_full = pd.read_csv("geocode_crosswalk.csv").drop(columns=["state"]).merge(
    pd.read_csv("geocode_output.csv", header=None,
                names=["uid", "input_address", "match_status", "match_type", "matched_address", "coords", "tid", "side"]),
    on="uid", how="left")
geo_full[["glon", "glat"]] = geo_full["coords"].str.split(",", expand=True).astype(float)
geo_full["_seq"] = geo_full.groupby("state_src").cumcount()
geo_full = geo_full.rename(columns={"state_src": "state"})[["state", "_seq", "match_status", "glat", "glon"]]

master = master.merge(geo_full, on=["state", "_seq"], how="left")

master["lat"] = master["lat_src"]
master["lon"] = master["lon_src"]
street_ok = master["match_status"] == "Match"
master.loc[master["lat"].isna() & street_ok, "lat"] = master.loc[master["lat"].isna() & street_ok, "glat"]
master.loc[master["lon"].isna() & street_ok, "lon"] = master.loc[master["lon"].isna() & street_ok, "glon"]
master["geocode_source"] = np.where(master["lat_src"].notna(), "state_api",
                             np.where(street_ok, "census_street", "unmatched"))

# ZIP-centroid fallback for anything still unmatched
zcta = pd.read_csv("zcta_gaz/2023_Gaz_zcta_national.txt", sep="\t")
zcta.columns = [c.strip() for c in zcta.columns]
zcta["GEOID"] = zcta["GEOID"].astype(str).str.zfill(5)
zcta_lookup = zcta.set_index("GEOID")[["INTPTLAT", "INTPTLONG"]]

need_fallback = master["lat"].isna() & master["zip5"].notna()
zc = master.loc[need_fallback, "zip5"].map(zcta_lookup["INTPTLAT"])
zc_lon = master.loc[need_fallback, "zip5"].map(zcta_lookup["INTPTLONG"])
master.loc[need_fallback, "lat"] = zc
master.loc[need_fallback, "lon"] = zc_lon
master.loc[need_fallback & master["lat"].notna(), "geocode_source"] = "zip_centroid"

master = master.drop(columns=["_seq", "lat_src", "lon_src", "glat", "glon", "match_status"])

print(master.shape)
print(master["geocode_source"].value_counts())
print(master.groupby("state")["geocode_source"].value_counts())
print("Still missing coords:", master["lat"].isna().sum())
print(master["date_confidence"].value_counts())
print("Missing open_date:", master["open_date"].isna().sum())

master["license_number"] = master["license_number"].astype(str)
master["address"] = master["address"].astype(str)

master.to_csv("dispensaries_master_geocoded.csv", index=False)
master.to_stata("dispensaries_master_geocoded.dta", write_index=False,
                 convert_dates={"open_date": "td"}, version=118)
