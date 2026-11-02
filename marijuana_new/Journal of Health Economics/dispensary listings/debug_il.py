import pdfplumber

with pdfplumber.open("il_raw.pdf") as pdf:
    for pageno in range(1, 21):
        page = pdf.pages[pageno]
        tables = page.extract_tables()
        total_rows = sum(len(t) for t in tables)
        print(pageno + 1, "tables:", len(tables), "total_rows:", total_rows)
