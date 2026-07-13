

from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]

STATIONS_FILE = ROOT / "data" / "cleaned" / "isd_history_station_check.csv"
OUTPUT_FILE = ROOT / "data" / "raw" / "NOAA_ISD_2021_2025.csv"

BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{usaf}{wban}.csv"
YEARS = range(2021, 2026)

HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}


def load_stations() -> pd.DataFrame:
    stations = pd.read_csv(STATIONS_FILE, dtype=str)
    stations = stations[stations["found"].astype(str).str.lower().isin(["true", "1"])]

    # USAF = 6 digits, WBAN = 5 digits, preserve leading zeros
    stations["usaf"] = stations["id"].astype(str).str.zfill(5) + "0"
    stations["wban"] = "99999"

    return stations[["icao", "usaf", "wban"]].dropna(subset=["usaf", "wban"])


def download_station_year(usaf: str, wban: str, year: int) -> pd.DataFrame | None:
    url = BASE_URL.format(year=year, usaf=usaf, wban=wban)

    for attempt in range(5):
        response = requests.get(url, headers=HEADERS, timeout=120)

        if response.status_code == 404:
            # No data for this station/year combination -> normal, skip it
            return None

        if response.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"  Rate limit reached, waiting {wait}s...")
            time.sleep(wait)
            continue

        response.raise_for_status()
        break
    else:
        raise RuntimeError(f"Failed to download {usaf}{wban} ({year}) after 5 attempts.")

    if not response.content.strip():
        return None

    df = pd.read_csv(StringIO(response.text), dtype=str, low_memory=False)
    return df if not df.empty else None


def main():
    stations = load_stations()
    all_frames = []
    found_stations = set()

    for _, row in stations.iterrows():
        icao, usaf, wban = row["icao"], row["usaf"], row["wban"]

        for year in YEARS:
            print(f"Downloading {icao} ({usaf}{wban}) - {year}...")
            df = download_station_year(usaf, wban, year)

            if df is not None:
                df["icao"] = icao
                all_frames.append(df)
                found_stations.add(icao)

            time.sleep(0.5)

    if not all_frames:
        raise RuntimeError("No data downloaded. Check the USAF/WBAN identifiers.")

    noaa = pd.concat(all_frames, ignore_index=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    noaa.to_csv(OUTPUT_FILE, index=False)

    print("\nDownload completed.")
    print(f"Stations with data: {len(found_stations)} / {len(stations)}")
    print(f"Rows: {len(noaa):,}")
    print(f"Saved to:\n{OUTPUT_FILE}")

    missing = sorted(set(stations["icao"]) - found_stations)
    if missing:
        print("\nStations with no data over 2021-2025:")
        print(missing)
    else:
        print("\nAll stations have at least some data.")


if __name__ == "__main__":
    main()