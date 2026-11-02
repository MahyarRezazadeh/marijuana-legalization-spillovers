import pdfplumber

with pdfplumber.open("il_raw.pdf") as pdf:
    page = pdf.pages[1]  # page 2
    tables = page.extract_tables()
    for i, t in enumerate(tables):
        print(f"--- table {i} ({len(t)} rows) ---")
        for r in t:
            print(r)
