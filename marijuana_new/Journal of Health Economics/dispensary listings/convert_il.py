import pandas as pd
import re

df = pd.read_csv("il_parsed_raw.csv")

def parse_addr(raw):
    raw = re.sub(r"\(\d{3}\)[\d\s\-\.]+$", "", raw).strip().rstrip(",")
    raw = re.sub(r",\s*,+", ",", raw)
    raw = re.sub(r"(\d{5})-,?\s*(\d{4})", r"\1-\2", raw)
    m = re.search(r"^(.*?),?\s*([A-Za-z .']+?),\s*(?:IL|Illinois)\.?,?\s*(\d{5})(?:-\d{4})?\s*$", raw)
    if m:
        return pd.Series([m.group(1).strip().rstrip(","), m.group(2).strip(), m.group(3)])
    return pd.Series([raw, None, None])

df[["address", "city", "zip"]] = df["address_raw"].apply(parse_addr)
zip_fallback = df["address_raw"].str.extract(r"(\d{5})(?:-\d{4})?")[0]
df["zip"] = df["zip"].fillna(zip_fallback)


def clean_name(s):
    s = re.sub(r",\s*(?=(LLC|Inc\.?|Corporation|Company)\b)", " ", s)
    return re.sub(r"\s*,\s*", " ", s).strip()


out = pd.DataFrame({
    "state": "IL",
    "license_number": df["credential_number"],
    "legal_name": df["license_holder"].apply(clean_name),
    "dba": df["dispensary_name"].apply(clean_name),
    "address": df["address"],
    "city": df["city"],
    "zip": df["zip"],
    "address_raw": df["address_raw"],
    "issuance_date": pd.to_datetime(df["issue_date"], errors="coerce"),
})

out.to_stata(
    "il_dispensaries.dta", write_index=False,
    convert_dates={"issuance_date": "td"},
    version=118,
)
print(out.shape)
print("Missing city/zip parse:", out["city"].isna().sum())
print(out.head(10).to_string())
