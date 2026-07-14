# This script orchestrates the entire METAR collection pipeline, which consists of four main steps:
# 1. Downloading NOAA ISD data (2021-2025) in CSV format.
# 2. Parsing the NOAA ISD data into a more usable METAR format, extracting core weather fields and enriching with cloud and present weather information using the avwx library.
# 3. Parsing the bulletin archive (metars_1.csv and metars_backup.csv) to extract METAR messages and their timestamps.
# 4. Merging the NOAA and bulletin sources into a single CSV file, keeping only the official ICAO stations as defined by the ISD station history mapping file, and deduplicating to ensure that there is only one observation per (icao, hour) pair, with NOAA observations taking priority over Bulletin observations in case of duplicates.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metar_collection.download_noaa_isd import main as download_noaa
from src.metar_collection.parse_noaa import main as parse_noaa
from src.metar_collection.parse_metars_history import main as parse_metars_history
from src.metar_collection.merge_metar_sources import main as merge_metar_sources


def main():
    print("=" * 70)
    print("STEP 1/4 - Downloading NOAA ISD data")
    print("=" * 70)
    download_noaa()

    print("\n")
    print("=" * 70)
    print("STEP 2/4 - Parsing NOAA METAR data")
    print("=" * 70)
    parse_noaa()

    print("\n")
    print("=" * 70)
    print("STEP 3/4 - Parsing bulletin archive (metars_1.csv, metars_backup.csv)")
    print("=" * 70)
    parse_metars_history()

    print("\n")
    print("=" * 70)
    print("STEP 4/4 - Merging NOAA and bulletin sources")
    print("=" * 70)
    merge_metar_sources()

    print("\n")
    print("=" * 70)
    print("METAR collection pipeline completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()