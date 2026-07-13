from pathlib import Path
import pandas as pd
import requests
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]

STATIONS_FILE = ROOT / "data" / "cleaned" / "isd_history_station_check.csv"
OUTPUT_FILE = ROOT / "data" / "raw" / "METAR_2021_2025.csv"

stations = pd.read_csv(STATIONS_FILE)
icaos = stations.loc[stations["found"], "icao"].dropna().tolist()

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

all_data = []

for icao in icaos:
    print(f"Downloading {icao}...")

    params = [
        ("station", icao),
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

    r = requests.get(BASE_URL, params=params, timeout=300)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    all_data.append(df)

metar = pd.concat(all_data, ignore_index=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
metar.to_csv(OUTPUT_FILE, index=False)

print(f"\nDownloaded {len(icaos)} stations.")
print(f"Total observations: {len(metar):,}")
print(f"Saved to:\n{OUTPUT_FILE}")