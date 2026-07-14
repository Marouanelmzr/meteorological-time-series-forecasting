# This script merges the NOAA and Bulletin METAR sources into a single CSV file, keeping only the official ICAO stations as defined by the ISD station history mapping file. 
# The merged output is deduplicated to ensure that there is only one observation per (icao, hour) pair,
#  with NOAA observations taking priority over Bulletin observations in case of duplicates.

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

NOAA_FILE = ROOT / "data" / "raw" / "METAR_2021_2025.csv"
BULLETIN_FILE = ROOT / "data" / "raw" / "METAR_bulletin_2021_2025.csv"
MAPPING_FILE = ROOT / "data" / "cleaned" / "isd_history_station_check.csv"
OUTPUT_FILE = ROOT / "data" / "raw" / "METAR_merged_sources_2021_2025.csv"


def load_official_icaos(path: Path) -> set:
    mapping = pd.read_csv(path, dtype=str)
    mapping = mapping[mapping["found"].astype(str).str.lower().isin(["true"])]
    return set(mapping["icao"].dropna())


def load_and_prepare(path: Path, source_label: str, official_icaos: set) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"icao": str, "station": str}, low_memory=False)
    df = df[df["icao"].isin(official_icaos)].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.floor("h")
    df["source"] = source_label

    # One row per (icao, hour): keep the observation closest to the top of the hour
    df["minute_offset"] = (df["date"] - df["hour"]).abs()
    df = (
        df.sort_values("minute_offset")
          .drop_duplicates(subset=["icao", "hour"], keep="first")
          .drop(columns=["minute_offset"])
    )
    return df


def main():
    official_icaos = load_official_icaos(MAPPING_FILE)
    print(f"Official stations: {len(official_icaos)}")

    noaa = load_and_prepare(NOAA_FILE, "noaa", official_icaos)
    bulletin = load_and_prepare(BULLETIN_FILE, "bulletin", official_icaos)

    print(f"NOAA hourly observations: {len(noaa):,}")
    print(f"Bulletin hourly observations: {len(bulletin):,}")

    noaa_keys = noaa[["icao", "hour"]].drop_duplicates()
    noaa_keys["_in_noaa"] = True
    bulletin_gap_fill = bulletin.merge(noaa_keys, on=["icao", "hour"], how="left")
    bulletin_gap_fill = bulletin_gap_fill[bulletin_gap_fill["_in_noaa"].isna()].drop(columns=["_in_noaa"])

    print(f"Bulletin rows used to fill NOAA gaps: {len(bulletin_gap_fill):,}")

    merged = pd.concat([noaa, bulletin_gap_fill], ignore_index=True)

    dup_mask = merged.duplicated(subset=["icao", "hour"], keep=False)
    n_dupes = dup_mask.sum()
    if n_dupes:
        print(f"WARNING: {n_dupes:,} duplicate (icao, hour) rows found -- deduping "
              f"(noaa priority, then closest to the hour).")
        merged["_source_rank"] = (merged["source"] != "noaa").astype(int)
        merged["_minute_offset"] = (merged["date"] - merged["hour"]).abs()
        merged = (
            merged.sort_values(["_source_rank", "_minute_offset"])
                  .drop_duplicates(subset=["icao", "hour"], keep="first")
                  .drop(columns=["_source_rank", "_minute_offset"])
        )
    else:
        print("No duplicate (icao, hour) pairs found.")

    merged = merged.drop(columns=["hour"]).sort_values(["icao", "date"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\nMerged total: {len(merged):,} rows")
    print(merged.groupby(["icao", "source"]).size().unstack(fill_value=0))
    print(f"\nSaved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()