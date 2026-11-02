import pandas as pd

df = pd.read_excel("ohio_dispensaries_raw.xlsx")

df.columns = [
    "license_number",
    "legal_name",
    "dba",
    "address",
    "city",
    "zip",
    "issuance_date",
    "expiration_date",
]

df["zip"] = df["zip"].astype(str).str.strip().str.slice(0, 5)
df["issuance_date"] = pd.to_datetime(df["issuance_date"])
df["expiration_date"] = pd.to_datetime(df["expiration_date"])
df.insert(0, "state", "OH")

df.to_stata(
    "ohio_dispensaries.dta",
    write_index=False,
    convert_dates={"issuance_date": "td", "expiration_date": "td"},
    version=118,
)

print(df.shape)
print(df.dtypes)
print(df.head(3).to_string())
