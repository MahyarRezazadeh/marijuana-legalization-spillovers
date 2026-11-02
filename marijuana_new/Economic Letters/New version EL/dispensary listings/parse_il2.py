import pdfplumber
import re
import pandas as pd

rows = []
settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}

with pdfplumber.open("il_raw.pdf") as pdf:
    for pageno in range(1, 21):
        page = pdf.pages[pageno]
        tables = page.extract_tables(table_settings=settings)
        for table in tables:
            for row in table:
                if row is None or len(row) < 5:
                    continue
                holder, name, addr, issue_date, credential = row[:5]
                if holder is None or "License Holder" in str(holder):
                    continue
                if not re.search(r"\d{1,2}/\d{1,2}/\d{4}", str(issue_date) or ""):
                    continue
                rows.append({
                    "license_holder": (holder or "").replace("\n", " ").strip(),
                    "dispensary_name": (name or "").replace("\n", " ").strip(),
                    "address_raw": (addr or "").replace("\n", ", ").strip(),
                    "issue_date": (issue_date or "").strip(),
                    "credential_number": (credential or "").strip(),
                })

print(f"Parsed {len(rows)} rows")
df = pd.DataFrame(rows).drop_duplicates(subset=["credential_number"])
print(f"After dedup: {df.shape}")
df.to_csv("il_parsed_raw.csv", index=False)
