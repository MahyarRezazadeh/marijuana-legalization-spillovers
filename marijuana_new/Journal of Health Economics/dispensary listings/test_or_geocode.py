import pandas as pd
import requests
import io

or_df = pd.read_stata("or_dispensaries.dta")
sample = or_df.head(20).copy()
sample["uid"] = range(1, len(sample) + 1)

buf = io.StringIO()
for _, r in sample.iterrows():
    buf.write(f'{r["uid"]},"{r["address"]}",,,\n')
content = buf.getvalue()
print(content[:300])

resp = requests.post(
    "https://geocoding.geo.census.gov/geocoder/locations/addressbatch",
    files={"addressFile": ("t.csv", content, "text/csv")},
    data={"benchmark": "Public_AR_Current"},
    timeout=120,
)
print(resp.text)
