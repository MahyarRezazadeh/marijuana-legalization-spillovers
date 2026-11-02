# ============================================================
# Alcohol placebo, Montana leg: Montana is a control state for spirits
# (private "agency liquor stores" under contract to the state, same
# ABCD division that publishes the cannabis CARD reports scraped
# elsewhere in this project). No bulk API exists -- each year's
# "Enterprise Fund Report of Operations" PDF has a page-by-page
# "Liquor Sales to Agency Stores by County" schedule, but the exact
# table layout changes across years (county-name-once-then-city-rows
# in FY18-20/22-23, pre-aggregated county rows in FY24). FY21's
# schedule is a scanned image, not extractable text -- skipped.
#
# Parsing heuristic (robust across all the layouts actually seen):
# within the text block between one recognized county name and the
# next, every dollar figure on its own token is either an individual
# store's sales or the county's own Total row -- and the Total (being
# a sum of positive store sales) is always the largest number in that
# block, even in the FY24 format where there's only one number to
# begin with. So: take the MAX dollar figure per county block as that
# county's annual total. Also count how many distinct "$" lines precede
# the max as a proxy store count where useful, but the reported
# "Number of Stores" column (FY24 only) is used directly when present.
# ============================================================
import os
import re
import fitz
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = r"C:\Users\mahya\AppData\Local\Temp\claude\C--Users-mahya\0b0891da-ef86-4c16-88d9-13084c23c680\scratchpad"
RAW = os.path.join(BASE, 'data', 'raw', 'placebo_alcohol')
os.makedirs(RAW, exist_ok=True)

MT_COUNTIES = [
    "Beaverhead", "Big Horn", "Blaine", "Broadwater", "Carbon", "Carter", "Cascade",
    "Chouteau", "Choteau", "Custer", "Daniels", "Dawson", "Deer Lodge", "Fallon", "Fergus",
    "Flathead", "Gallatin", "Garfield", "Glacier", "Golden Valley", "Granite", "Hill",
    "Jefferson", "Judith Basin", "Lake", "Lewis and Clark", "Liberty", "Lincoln",
    "Madison", "McCone", "Meagher", "Mineral", "Missoula", "Musselshell", "Park",
    "Petroleum", "Phillips", "Pondera", "Powder River", "Powell", "Prairie",
    "Ravalli", "Richland", "Roosevelt", "Rosebud", "Sanders", "Sheridan",
    "Silver Bow", "Stillwater", "Sweet Grass", "Sweetgrass", "Teton", "Toole",
    "Treasure", "Valley", "Wheatland", "Wibaux", "Yellowstone",
]
# sort longest-first so "Lewis and Clark" matches before a hypothetical shorter prefix
MT_COUNTIES_SORTED = sorted(set(MT_COUNTIES), key=len, reverse=True)

FY_YEARS = [2018, 2019, 2020, 2022, 2023, 2024]


def find_county_pages(doc):
    """Auto-detect pages belonging to the by-county schedule, rather than
    guessing a page range -- guessing previously pulled in a spillover page
    from the by-STORE ranking table (Table 28) for FY2024, double-counting
    every dollar figure on it under whichever county block happened to be
    open at that point."""
    pages = []
    for i in range(len(doc)):
        t = doc[i].get_text()
        has_by_county_header = (
            "Schedule of Liquor Sales by County" in t
            or "SALES TO AGENCY \nSTORES BY COUNTY" in t
            or "SALES TO AGENCY\nSTORES BY COUNTY" in t
            or "SALES TO AGENCY \nSTORE BY COUNTY" in t
        )
        # FY24's format has its own distinct header + a "Number of Stores" column
        # that never appears on the by-store ranking pages -- use it to
        # disambiguate rather than page position.
        if has_by_county_header and len(t.strip()) > 150:
            pages.append(i)
    return pages


def parse_year(fy):
    path = os.path.join(SCRATCH, f"mt_liquor_fy{fy}.pdf")
    doc = fitz.open(path)
    pages = find_county_pages(doc)
    print(f"FY{fy}: county-schedule pages detected = {pages}")
    text = ""
    for i in pages:
        text += "\n" + doc[i].get_text()

    # tokenize into lines, drop boilerplate
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    drop_patterns = [
        "BACK TO TABLE OF CONTENTS", "LIQUOR SALES TO AGENCY", "STORES BY COUNTY",
        "STORE BY COUNTY", "Schedule of Liquor Sales by County",
        "For the fiscal year ending", "Gross", "Percent of", "County", "City",
        "Sales", "Total Sales", "Number of Stores", "Gross Sales to Agent",
        "Percent of Sales",
    ]
    clean = []
    for l in lines:
        if l in ("County", "City", "Sales", "Total Sales", "Gross", "Percent of Sales",
                  "Gross Sales to Agent", "Number of Stores"):
            continue
        if any(p in l for p in ["BACK TO TABLE OF CONTENTS", "LIQUOR SALES TO AGENCY",
                                 "STORES BY COUNTY", "STORE BY COUNTY",
                                 "Schedule of Liquor Sales by County",
                                 "For the fiscal year ending"]):
            continue
        if re.fullmatch(r"\d{1,3}", l):  # bare page numbers
            continue
        clean.append(l)

    # Drop the statewide grand-total row and everything after it. Without
    # this, the LAST county in the table (Yellowstone, alphabetically) has no
    # following county name to close its block, so it absorbs the grand-total
    # line too -- and since the grand total is bigger than any single
    # county's figure, the max-per-block heuristic below picks the grand
    # total instead of Yellowstone's real number. Every format uses a
    # standalone "Total"/"Totals" line for both per-county subtotals AND the
    # one statewide grand total, but the grand total is always the LAST such
    # standalone marker in the table -- so cut there.
    total_marker_idxs = [idx for idx, l in enumerate(clean) if re.fullmatch(r"(?i)totals?", l)]
    if total_marker_idxs:
        clean = clean[:total_marker_idxs[-1]]

    # find county-name line starts (line == county name, or startswith county name).
    # Reject false matches where a STORE happens to be named after a county
    # (e.g. store "Sheridan #105" sitting inside Madison county's own block,
    # or "Deer Lodge #11" inside Powell's block -- both real MT town/county
    # name collisions seen in this data): a genuine header is either the
    # whole line, or "County Cityname #NNN" (a real word between county name
    # and the store number); a collision is "County #NNN" with nothing
    # in between, because the "county name" IS the store's city name.
    collision_re = re.compile(r"^\s*#\d")
    CANONICAL = {"Choteau": "Chouteau"}  # FY2024 report misspells Chouteau County

    def match_county(line):
        for cname in MT_COUNTIES_SORTED:
            if line == cname:
                return CANONICAL.get(cname, cname)
            for sep in (" ", "\xa0"):
                if line.startswith(cname + sep):
                    remainder = line[len(cname) + 1:]
                    if collision_re.match(remainder):
                        return None  # false match, e.g. "Sheridan #105"
                    return CANONICAL.get(cname, cname)
        return None

    county_starts = []
    for idx, l in enumerate(clean):
        cname = match_county(l)
        if cname:
            county_starts.append((idx, cname))

    money_re = re.compile(r"\$?\s*([\d,]+\.\d{2}|\d{1,3}(?:,\d{3})+|\d{4,})\s*$")

    def extract_money(s):
        s = s.replace("\xa0", " ").strip()
        m = re.search(r"\$?\s*([\d,]+(?:\.\d{2})?)", s)
        if not m:
            return None
        val = m.group(1).replace(",", "")
        try:
            f = float(val)
        except ValueError:
            return None
        return f

    results = {}
    for k, (idx, cname) in enumerate(county_starts):
        end = county_starts[k + 1][0] if k + 1 < len(county_starts) else len(clean)
        block = clean[idx:end]
        # strip the county name prefix off the first line, keep remainder as first token
        first = block[0]
        for pref in (cname + " ", cname + "\xa0"):
            if first.startswith(pref):
                block[0] = first[len(pref):].strip()
                break
        else:
            block[0] = ""
        # Prefer the county's own "Total"/"Totals" marker line (the $ figure
        # immediately after it) over a blind max() of the whole block.
        # Two-column PDF layouts can interleave a NEIGHBORING county's tail
        # end into this block's text stream out of visual order (seen in
        # FY2022: Yellowstone's own store rows got extracted right after
        # Fergus's genuine Total line, so a blind max() wrongly gave Fergus
        # Yellowstone's much larger total). Stopping at the first Total
        # marker ignores anything appended after it within the same block.
        total_idx = next((i for i, line in enumerate(block)
                           if re.fullmatch(r"(?i)totals?", line)), None)
        if total_idx is not None:
            total = None
            for line in block[total_idx + 1:]:
                v = extract_money(line)
                if v is not None and v > 0:
                    total = v
                    break
        else:
            vals = [v for line in block if (v := extract_money(line)) is not None and v > 0]
            total = max(vals) if vals else None
        if total is None:
            continue
        results[cname] = max(results.get(cname, 0), total)

    rows = [{"county": c, "fy_year": fy, "total_sales": v} for c, v in results.items()]
    out = pd.DataFrame(rows)
    print(f"FY{fy}: parsed {len(out)} counties, sum=${out['total_sales'].sum():,.0f}")
    return out


all_years = []
for fy in FY_YEARS:
    all_years.append(parse_year(fy))

panel = pd.concat(all_years, ignore_index=True)
panel.to_csv(os.path.join(RAW, 'mt_liquor_county_fy_raw.csv'), index=False)
print(f"\nSaved {len(panel)} county-year rows to {os.path.join(RAW, 'mt_liquor_county_fy_raw.csv')}")
print(f"Counties per year:\n{panel.groupby('fy_year')['county'].nunique()}")
