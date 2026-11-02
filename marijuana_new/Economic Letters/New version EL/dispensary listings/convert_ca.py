import json
import pandas as pd

rows = []
for f in ["ca_raw.json", "ca_raw_p2.json"]:
    with open(f, encoding="utf-8") as fh:
        rows.extend(json.load(fh)["data"])

df = pd.DataFrame(rows)
print("Total rows:", df.shape[0])
print(df["licenseStatus"].value_counts())

out = pd.DataFrame({
    "state": "CA",
    "license_number": df["licenseNumber"],
    "legal_name": df["businessLegalName"],
    "dba": df["businessDbaName"],
    "address": df["premiseStreetAddress"],
    "city": df["premiseCity"],
    "county": df["premiseCounty"],
    "zip": df["premiseZipCode"].astype(str).str.slice(0, 5),
    "latitude": df["premiseLatitude"],
    "longitude": df["premiseLongitude"],
    "license_status": df["licenseStatus"],
    "issuance_date": pd.to_datetime(df["issueDate"], errors="coerce"),
    "expiration_date": pd.to_datetime(df["expirationDate"], errors="coerce"),
})
out.to_stata(
    "ca_dispensaries.dta", write_index=False,
    convert_dates={"issuance_date": "td", "expiration_date": "td"},
    version=118,
)
print(out.shape)
print("Missing lat/lon:", out["latitude"].isna().sum())
