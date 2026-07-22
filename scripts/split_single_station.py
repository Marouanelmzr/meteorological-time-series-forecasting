# For single station split ( follows same logic as split_dataset.py )

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Arguments

parser = argparse.ArgumentParser(
    description="Create train/val/test splits for a group of stations."
)

parser.add_argument(
    "--stations",
    nargs="+",
    required=True,
    help="List of ICAO stations (e.g. GMMI GMME GMTT)",
)

args = parser.parse_args()

STATIONS = [s.upper() for s in args.stations]

# Paths

INPUT = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "FINAL_PREPARED_DATA_2021_2025.csv"
)

group_name = "_".join(s.lower() for s in STATIONS)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / f"splits_{group_name}"
)

TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load

df = pd.read_csv(
    INPUT,
    parse_dates=["run_time", "valid_time"],
)

print(f"Loaded {len(df):,} samples")


# Keep only selected station

df = df[df["icao"].isin(STATIONS)].copy()

if df.empty:
    raise ValueError(
        f"No samples found for stations {STATIONS}."
    )


print("Keeping stations:")
for station in STATIONS:
    print(f"  - {station}")

print(f"\nTotal samples : {len(df):,}")
print(f"Total gusts   : {int(df['has_gust'].sum()):,}")



# Assign split (using run_time only)

df["split"] = "test"

df.loc[df["run_time"] < VAL_END, "split"] = "val"
df.loc[df["run_time"] < TRAIN_END, "split"] = "train"


# Safety check

leakage = (
    df.groupby("run_time")["split"]
    .nunique()
)

assert leakage.max() == 1, "Data leakage detected between splits."


# Save splits

sort_cols = ["run_time", "valid_time"]

for split in ["train", "val", "test"]:

    subset = (
        df[df["split"] == split]
        .drop(columns="split")
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    subset.to_parquet(
        OUTPUT_DIR / f"{split}.parquet",
        index=False,
    )

    gusts = int(subset["has_gust"].sum())

    print(
        f"{split.upper():5s}"
        f" | samples={len(subset):7,d}"
        f" | gusts={gusts:5,d}"
        f" | {subset['run_time'].min().date()} -> {subset['run_time'].max().date()}"
    )

print("\nSplit completed successfully.")