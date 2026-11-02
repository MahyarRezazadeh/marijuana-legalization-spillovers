"""Clean WA county sales/excise-tax-by-county workbooks (FY15-FY25) -> long county-fiscal-year panel."""
import openpyxl
import pandas as pd

rows = []

def read_sheet(path, sheet, fy_year):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    header_row = None
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if row[0] == "County":
            header_row = i
            continue
        if header_row is None:
            continue
        county, period, total_sales, excise_tax = row[0], row[1], row[2], row[3]
        if county is None or total_sales is None:
            continue
        rows.append({
            "county_key": str(county).strip().upper(),
            "fy_year": fy_year,
            "total_sales": total_sales,
            "excise_tax": excise_tax,
        })

read_sheet("data/raw/wa_fy15.xlsx", "Retailer", 2015)
read_sheet("data/raw/wa_fy16.xlsx", "Retailer", 2016)
read_sheet("data/raw/wa_fy17.xlsx", "Retailer", 2017)
for fy in range(2018, 2025):
    read_sheet("data/raw/wa_fy2018_2024.xlsx", f"FY{fy % 100}", fy)
read_sheet("data/raw/wa_fy25.xlsx", "FY25", 2025)

df = pd.DataFrame(rows)
df["state_abbr"] = "WA"
df.to_csv("data/wa_revenue_annual.csv", index=False)
print(f"WA annual county-year rows: {len(df)}, years: {sorted(df['fy_year'].unique())}, counties: {df['county_key'].nunique()}")
