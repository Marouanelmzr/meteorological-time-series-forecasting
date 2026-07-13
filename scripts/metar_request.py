from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

STATIONS_FILE = ROOT / "data" / "cleaned" / "isd_history_station_check.csv"
OUTPUT_FILE = ROOT / "data" / "raw" / "METAR_2021_2025.csv"

# Load validated stations
stations = pd.read_csv(STATIONS_FILE)

# Keep only stations found in NOAA
icaos = stations.loc[stations["found"], "icao"].dropna().tolist()

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

params = [
    ("data", "all"),
    ("tz", "UTC"),
    ("format", "onlycomma"),
    ("latlon", "yes"),
    ("elev", "yes"),
    ("missing", "M"),
    ("trace", "T"),
    ("report_type", "1"),
    ("report_type", "2"),

    ("year1", 2021),
    ("month1", 1),
    ("day1", 1),

    ("year2", 2025),
    ("month2", 12),
    ("day2", 31),
]

# Add every ICAO station
for icao in icaos:
    params.append(("station", icao))

print(f"Downloading METAR data for {len(icaos)} stations...")

response = requests.get(BASE_URL, params=params, timeout=300)
response.raise_for_status()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(response.text)

print(f"Saved to:\n{OUTPUT_FILE}")