import requests

url = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
with open("geocode_input.csv", "rb") as f:
    files = {"addressFile": ("geocode_input.csv", f, "text/csv")}
    data = {"benchmark": "Public_AR_Current"}
    resp = requests.post(url, files=files, data=data, timeout=300)

print("Status:", resp.status_code)
with open("geocode_output.csv", "wb") as f:
    f.write(resp.content)
print("Bytes:", len(resp.content))
print(resp.text[:500])
