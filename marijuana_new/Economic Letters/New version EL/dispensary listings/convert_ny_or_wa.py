import pandas as pd

# ---------------- NY ----------------
ny = pd.read_csv("ny_raw.csv", low_memory=False)
retail_types = [
    "Adult-Use Retail Dispensary License",
    "Adult-Use Conditional Retail Dispensary License",
    "Adult-Use Registered Organization Dispensary License",
]
ny = ny[ny["license_type"].isin(retail_types)].copy()
ny_out = pd.DataFrame({
    "state": "NY",
    "license_number": ny["license_number"],
    "legal_name": ny["entity_name"],
    "dba": ny["dba"],
    "address": ny["address_line_1"],
    "city": ny["city"],
    "county": ny["county"],
    "zip": ny["zip_code"].astype(str).str.slice(0, 5),
    "license_type": ny["license_type"],
    "license_status": ny["license_status"],
    "issuance_date": pd.to_datetime(ny["issued_date"], errors="coerce"),
    "effective_date": pd.to_datetime(ny["effective_date"], errors="coerce"),
    "expiration_date": pd.to_datetime(ny["expiration_date"], errors="coerce"),
    "retail_date_opened_to_public": pd.to_datetime(ny["retail_date_opened_to_public"], errors="coerce"),
})
ny_out.to_stata(
    "ny_dispensaries.dta", write_index=False,
    convert_dates={c: "td" for c in ["issuance_date", "effective_date", "expiration_date", "retail_date_opened_to_public"]},
    version=118,
)
print("NY:", ny_out.shape)

# ---------------- OR ----------------
orf = pd.read_csv("or_raw.csv", low_memory=False)
orf = orf[orf["license_type"] == "RECREATIONAL RETAILER"].copy()
or_out = pd.DataFrame({
    "state": "OR",
    "license_number": orf["license_number"],
    "legal_name": orf["business_name"],
    "address": orf["physical_address"],
    "county": orf["county"],
    "license_type": orf["license_type"],
    "license_expired": orf["license_expired"],
    "effective_date": pd.to_datetime(orf["effective_date"], errors="coerce"),
    "expiration_date": pd.to_datetime(orf["expiration_date"], errors="coerce"),
    "inactive_date": pd.to_datetime(orf["inactive_date"], errors="coerce"),
})
or_out.to_stata(
    "or_dispensaries.dta", write_index=False,
    convert_dates={c: "td" for c in ["effective_date", "expiration_date", "inactive_date"]},
    version=118,
)
print("OR:", or_out.shape)

# ---------------- WA ----------------
wa = pd.read_csv("wa_raw.csv", low_memory=False)
priv_cols = [c for c in wa.columns if c.startswith("privdesc")]
is_retail = wa[priv_cols].apply(lambda row: row.astype(str).str.contains("Retailer", na=False).any(), axis=1)
wa = wa[is_retail].copy()
wa_out = pd.DataFrame({
    "state": "WA",
    "license_number": wa["license"],
    "legal_name": wa["tradename"],
    "address": wa["streetaddress"],
    "city": wa["city"],
    "county": wa["countyname"],
    "zip": wa["zipcode"].astype(str).str.slice(0, 5),
    "license_type": "Cannabis Retailer",
    "renewal_date": pd.to_datetime(wa["renewaldate"], format="%Y%m%d", errors="coerce"),
})
wa_out.to_stata(
    "wa_dispensaries.dta", write_index=False,
    convert_dates={"renewal_date": "td"},
    version=118,
)
print("WA:", wa_out.shape)
print("NOTE: WA and OR license_expired/inactive fields are renewal/expiration-side dates, not true opening dates like OH/NY/IL have.")
