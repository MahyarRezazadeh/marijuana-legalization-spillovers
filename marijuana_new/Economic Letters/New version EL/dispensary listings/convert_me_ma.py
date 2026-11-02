import pandas as pd

# ---------------- ME ----------------
me = pd.read_csv("me_raw.csv", low_memory=False)
me = me[me["LICENSE_CATEGORY"] == "Store"].drop_duplicates(subset=["LICENSE"]).copy()
me_out = pd.DataFrame({
    "state": "ME",
    "license_number": me["LICENSE"],
    "legal_name": me["LICENSE_NAME"],
    "dba": me["DBA"],
    "address": me["LICENSE_ADDRESS"],
    "city": me["LICENSE_CITY"],
    "county": me["LICENSE_COUNTY"],
    "license_status": me["LICENSE_STATUS"],
    "issuance_date": pd.to_datetime(me["LICENSE_ISSUE_DATE"], errors="coerce"),
    "first_issue_date": pd.to_datetime(me["LICENSE_FIRST_ISSUE_DATE"], errors="coerce"),
    "expiration_date": pd.to_datetime(me["LICENSE_EXPIRATION_DATE"], errors="coerce"),
})
me_out.to_stata(
    "me_dispensaries.dta", write_index=False,
    convert_dates={c: "td" for c in ["issuance_date", "first_issue_date", "expiration_date"]},
    version=118,
)
print("ME:", me_out.shape)

# ---------------- MA ----------------
ma = pd.read_csv("ma_raw.csv", low_memory=False)
ma = ma[(ma["LICENSE_TYPE"] == "Marijuana Retailer") & (ma["INDUSTRY"] == "Adult-Use")].copy()
ma_out = pd.DataFrame({
    "state": "MA",
    "license_number": ma["LICENSE_NUMBER"],
    "legal_name": ma["BUSINESS_NAME"],
    "address": ma["ESTABLISHMENT_ADDRESS_1"],
    "city": ma["ESTABLISHMENT_CITY"],
    "county": ma["ESTABLISHMENT_COUNTY"],
    "zip": ma["ESTABLISHMENT_ZIP"],
    "license_status": ma["LICENSE_STATUS"],
    "commence_operations_date": pd.to_datetime(ma["COMMENCE_OPERATIONS_DATE"], errors="coerce"),
    "final_licensure_date": pd.to_datetime(ma["CNB_DATE_OF_FINAL_LICENSURE"], errors="coerce"),
    "license_start_date": pd.to_datetime(ma["LIC_START_DATE"], errors="coerce"),
    "license_expiration_date": pd.to_datetime(ma["LIC_EXPIRATION_DATE"], errors="coerce"),
})
ma_out.to_stata(
    "ma_dispensaries.dta", write_index=False,
    convert_dates={c: "td" for c in ["commence_operations_date", "final_licensure_date", "license_start_date", "license_expiration_date"]},
    version=118,
)
print("MA:", ma_out.shape)
