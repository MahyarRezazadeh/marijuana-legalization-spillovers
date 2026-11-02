import pdfplumber
import re
import pandas as pd

DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
CRED_RE = re.compile(r"284[.\-]?\d+.*AUDO", re.IGNORECASE)

records = []

with pdfplumber.open("il_raw.pdf") as pdf:
    for pageno in range(1, 21):
        page = pdf.pages[pageno]
        tables = page.extract_tables()
        if not tables:
            continue
        table = max(tables, key=len)  # the real full table has the most rows
        current = None
        for row in table:
            cells = [re.sub(r",?\s*,", ",", c.replace("\n", ", ")).strip() if c else "" for c in row]
            holder = cells[0] if cells else ""
            if "License Holder" in holder:
                continue
            others = []
            date_val = None
            cred_val = None
            for c in cells[1:]:
                if not c:
                    continue
                if DATE_RE.match(c):
                    date_val = c
                elif CRED_RE.search(c):
                    cred_val = c
                else:
                    others.append(c)
            if holder:
                if current:
                    records.append(current)
                current = {
                    "license_holder": holder,
                    "dispensary_name": others[0] if others else "",
                    "address_parts": others[1:],
                    "issue_date": date_val or "",
                    "credential_number": cred_val or "",
                }
            else:
                if current is None:
                    continue
                current["address_parts"].extend(others)
                if date_val:
                    current["issue_date"] = date_val
                if cred_val:
                    current["credential_number"] = cred_val
        if current:
            records.append(current)

for r in records:
    r["address_raw"] = ", ".join(r.pop("address_parts"))

df = pd.DataFrame(records)
df = df[df["issue_date"] != ""]
df = df.drop_duplicates(subset=["credential_number"])
print(f"Parsed {df.shape[0]} rows")
print(df.tail(15).to_string())
df.to_csv("il_parsed_raw.csv", index=False)
