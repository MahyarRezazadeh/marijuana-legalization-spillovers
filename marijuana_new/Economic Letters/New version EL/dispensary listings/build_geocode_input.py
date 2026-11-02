import pandas as pd

files = {
    "OH": "ohio_dispensaries.dta",
    "NY": "ny_dispensaries.dta",
    "MA": "ma_dispensaries.dta",
    "OR": "or_dispensaries.dta",
    "WA": "wa_dispensaries.dta",
    "IL": "il_dispensaries.dta",
    "ME": "me_dispensaries.dta",
    "CO": "co_dispensaries.dta",
}

rows = []
uid = 0
for state, f in files.items():
    df = pd.read_stata(f)
    for i, r in df.iterrows():
        uid += 1
        addr = r.get("address", "") or ""
        city = r.get("city", "") or ""
        zip_ = r.get("zip", "") or ""
        # OR addresses are one undelimited blob (street+city+state+zip run together,
        # e.g. "10920 SW BEAVERTON HILLSDALE HWY BEAVERTON OR 97005-3003") with no
        # separate city/zip fields available. Passing "OR" again as a separate state
        # field conflicts with the state token already embedded in the blob and makes
        # the batch geocoder fail almost every row (tested: 26/981 matched). Leaving
        # city/state/zip blank and letting the geocoder parse the whole blob as
        # "street" fixes this (tested: 19/20 matched on a sample).
        state_field = "" if state == "OR" else state
        rows.append({
            "uid": uid,
            "state_src": state,
            "row_idx": i,
            "street": str(addr).strip(),
            "city": str(city).strip(),
            "state": state_field,
            "zip": str(zip_).strip(),
        })

geo_in = pd.DataFrame(rows)
geo_in.to_csv("geocode_crosswalk.csv", index=False)

# Census batch geocoder input: Unique ID, Street address, City, State, ZIP -- no header
batch = geo_in[["uid", "street", "city", "state", "zip"]]
batch.to_csv("geocode_input.csv", index=False, header=False)
print("Total addresses:", len(batch))
print(batch.head())
