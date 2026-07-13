# This script merges the AROME dataset with the METAR dataset, using the station mapping to match ICAO codes to WMO IDs. 
# The merged dataset is saved to a CSV file for further analysis.

import pandas as pd
import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(PROJECT_ROOT))

AROME_PATH = PROJECT_ROOT / "data" / "cleaned" / "Arome_clean_final.csv"
METAR_PATH = PROJECT_ROOT / "data" / "raw" / "METAR_2021_2025.csv"
MAPPING_PATH = PROJECT_ROOT / "data" / "cleaned" / "isd_history_station_check.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "AROME_METAR_merged_2021_2025.csv"

START_DATE = "2021-01-01"
END_DATE = "2025-12-31 23:59"


def load_station_mapping(path: Path) -> dict:
    mapping_df = pd.read_csv(path, dtype=str)
    mapping_df = mapping_df[mapping_df["found"] == "True"]
    icao_to_wmo = dict(zip(mapping_df["icao"], mapping_df["id"]))
    return icao_to_wmo


def load_arome(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H")

    before = len(df)
    df = df[(df["datetime"] >= START_DATE) & (df["datetime"] <= END_DATE)]
    print(f"AROME: {before} rows loaded, {len(df)} kept after restricting to "
          f"{START_DATE} -> {END_DATE} ({before - len(df)} dropped, mostly the 2019-2020 tail).")

    return df


def load_metar(path: Path, icao_to_wmo: dict) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station": str}, low_memory=False)

    df["valid"] = pd.to_datetime(df["valid"])
    df["datetime"] = df["valid"].dt.floor("h")

    df["wmo_id"] = df["station"].map(icao_to_wmo)

    unmatched = df[df["wmo_id"].isna()]["station"].unique()
    if len(unmatched):
        print( f"WARNING: {len(unmatched)} ICAO code(s) in METAR have no entry in the station mapping and will be dropped: {sorted(unmatched)}" )

    df = df.dropna(subset=["wmo_id"])

    before = len(df)

    # Keep one METAR per station per hour
    df = (
        df.sort_values("valid")
          .drop_duplicates(subset=["wmo_id", "datetime"], keep="first")
    )

    print(f"METAR: {before} rows -> {len(df)} hourly observations ({before - len(df)} duplicates removed).")

    return df


def merge(arome: pd.DataFrame, metar: pd.DataFrame) -> pd.DataFrame:
    # lon/lat are already present on both sides; the AROME grid-point coordinate
    # is kept as the reference feature, METAR's is kept alongside for QC only.
    merged = arome.merge(
        metar,
        left_on=["id", "datetime"],
        right_on=["wmo_id", "datetime"],
        how="inner",
        suffixes=("_arome", "_metar"),
    )

    n_arome, n_metar, n_merged = len(arome), len(metar), len(merged)
    print(f"\nAROME rows           : {n_arome}")
    print(f"METAR rows (matched station) : {n_metar}")
    print(f"Merged rows (inner join)     : {n_merged}")
    print(f"AROME rows without a METAR match : {n_arome - n_merged} "
          f"({(n_arome - n_merged) / n_arome:.1%})")

    return merged


def main():
    icao_to_wmo = load_station_mapping(MAPPING_PATH)
    arome = load_arome(AROME_PATH)
    metar = load_metar(METAR_PATH, icao_to_wmo)
    merged = merge(arome, metar)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved merged dataset to {OUTPUT_PATH}  ({merged.shape[0]} rows, {merged.shape[1]} columns)")


if __name__ == "__main__":
    main()