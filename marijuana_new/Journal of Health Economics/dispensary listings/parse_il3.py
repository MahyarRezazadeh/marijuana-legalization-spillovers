import pdfplumber
import re
import pandas as pd

records = []

with pdfplumber.open("il_raw.pdf") as pdf:
    for pageno in range(1, 21):  # pages 2-21 (0-indexed 1-20)
        page = pdf.pages[pageno]
        tables = page.extract_tables()
        if not tables:
            continue
        table = tables[0]  # the real table is always the first/largest one
        current = None
        for row in table:
            row = list(row) + [None] * (9 - len(row))
            holder, name, name2, _, addr4, addr5, addr6, issue_date, credential = row[:9]
            if holder and "License Holder" in str(holder):
                continue
            if holder:
                if current:
                    records.append(current)
                current = {
                    "license_holder": holder.replace("\n", " ").strip(),
                    "dispensary_name": (name or "").replace("\n", " ").strip(),
                    "address_parts": [],
                    "issue_date": (issue_date or "").strip(),
                    "credential_number": (credential or "").strip(),
                }
                for extra in (name2,):
                    if extra:
                        current["dispensary_name"] += " " + extra.replace("\n", " ").strip()
                for a in (addr4, addr5, addr6):
                    if a:
                        current["address_parts"].append(a.replace("\n", ", ").strip())
            else:
                if current is None:
                    continue
                for extra in (name2,):
                    if extra:
                        current["dispensary_name"] += " " + extra.replace("\n", " ").strip()
                for a in (addr4, addr5, addr6):
                    if a:
                        current["address_parts"].append(a.replace("\n", ", ").strip())
                if issue_date:
                    current["issue_date"] = issue_date.strip()
                if credential:
                    current["credential_number"] = credential.strip()
        if current:
            records.append(current)

for r in records:
    r["address_raw"] = ", ".join(r.pop("address_parts"))

df = pd.DataFrame(records)
df = df[df["issue_date"].str.contains(r"\d{1,2}/\d{1,2}/\d{4}", na=False, regex=True)]
df = df.drop_duplicates(subset=["credential_number"])
print(f"Parsed {df.shape[0]} rows")
print(df.head(15).to_string())
df.to_csv("il_parsed_raw.csv", index=False)
