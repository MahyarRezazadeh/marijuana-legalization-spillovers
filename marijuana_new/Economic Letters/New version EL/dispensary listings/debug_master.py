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

master = []
for state, (fname, datecol) in files.items():
    df = pd.read_stata(fname)
    df["state"] = state
    df["open_date"] = pd.Timestamp("2015-01-01")
    df["date_confidence"] = "x"
    zip_col = df["zip"] if "zip" in df.columns else df["address"]
    df["zip5"] = zip_col.astype(str).str.extract(r"(\d{5})")[0]
    keep = ["state", "license_number", "address", "zip5", "open_date", "date_confidence"]
    df["lat_src"] = np.nan
    df["lon_src"] = np.nan
    keep += ["lat_src", "lon_src"]
    sub = df[keep]
    print(state, sub.columns.tolist(), sub.columns.duplicated().any())
    master.append(sub)

m = pd.concat(master, ignore_index=True)
print("FINAL DUPES:", m.columns[m.columns.duplicated()].tolist())
