"""Clean CO monthly county marijuana sales -> county-calendar-year totals."""
import pandas as pd

df = pd.read_csv("data/raw/co_marijuana_sales_by_county.csv")

df = df[df["county"] != "Total"].copy()
df["year"] = df["year"].astype(float).astype(int)
df["month"] = df["month"].astype(int)
df["med_sales"] = pd.to_numeric(df["med_sales"], errors="coerce").fillna(0)
df["rec_sales"] = pd.to_numeric(df["rec_sales"], errors="coerce").fillna(0)
df["total_sales"] = df["med_sales"] + df["rec_sales"]

# calendar-year totals per county; require all 12 months present for a clean annual total
counts = df.groupby(["county", "year"]).size().rename("n_months")
annual = df.groupby(["county", "year"])["total_sales"].sum().reset_index()
annual = annual.merge(counts, on=["county", "year"])
annual = annual[annual["n_months"] == 12].drop(columns="n_months")

annual["county_key"] = annual["county"].str.strip().str.upper()
annual["state_abbr"] = "CO"
annual = annual.rename(columns={"year": "fy_year"})

annual.to_csv("data/co_revenue_annual.csv", index=False)
print(f"CO annual county-year rows: {len(annual)}, years: {sorted(annual['fy_year'].unique())}, counties: {annual['county_key'].nunique()}")
