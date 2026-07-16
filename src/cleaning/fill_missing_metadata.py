"""
Fill missing station metadata for GMFI (Ifrane Airport).

This script fills missing values of:
- station_lat
- station_lon
- elevation_m

Only NaN values are replaced; existing values are never overwritten.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025.csv"

GMFI_METADATA = {
    "station_lat": 33.5053,
    "station_lon": -5.1529,
    "elevation_m": 1664.0,
}


def main():
    df = pd.read_csv(INPUT_FILE)

    gmfi = df["icao"] == "GMFI"

    for column, value in GMFI_METADATA.items():
        df.loc[gmfi & df[column].isna(), column] = value

    df.to_csv(INPUT_FILE, index=False)

    print("GMFI station metadata successfully updated.")


if __name__ == "__main__":
    main()