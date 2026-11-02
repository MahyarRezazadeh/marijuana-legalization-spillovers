import pandas as pd

co = pd.read_csv("co_raw.csv")
co_out = pd.DataFrame({
    "state": "CO",
    "license_number": co["License Number"],
    "legal_name": co["Facility Name"],
    "dba": co["DBA"],
    "address": co["Street"],
    "city": co["City"],
    "zip": co["ZIP Code"].astype(str).str.slice(0, 5),
    "expiration_date": pd.to_datetime(co["Expiration Date"], errors="coerce"),
    "date_updated": pd.to_datetime(co["Date Updated"], errors="coerce"),
})
co_out.to_stata(
    "co_dispensaries.dta", write_index=False,
    convert_dates={"expiration_date": "td", "date_updated": "td"},
    version=118,
)
print("CO:", co_out.shape)
