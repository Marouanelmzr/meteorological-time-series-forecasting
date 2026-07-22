"""
Split AROME_METAR dataset into train / validation / test.

Split strategy
--------------
- Split chronologically using run_time (never valid_time).
- Train : run_time < 2024-01-01
- Val   : 2024-01-01 <= run_time < 2025-01-01
- Test  : run_time >= 2025-01-01

Notes
-----
- Splitting on run_time prevents data leakage between lead times of the same
  forecast cycle.
- GSVO is removed only from the test split because almost no observations
  remain after 2022.
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(PROJECT_ROOT))


# Paths

# INPUT = PROJECT_ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025.csv"
INPUT = PROJECT_ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025_with_rolling_features.csv"
# OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits_rolling_features"

TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load

df = pd.read_csv(
    INPUT,
    parse_dates=["run_time", "valid_time"],
)

print(f"Loaded {len(df):,} samples")

# Assign split (using run_time only)

df["split"] = "test"

df.loc[df["run_time"] < VAL_END, "split"] = "val"
df.loc[df["run_time"] < TRAIN_END, "split"] = "train"

# Safety check : one run_time must belong to only one split

leakage = (
    df.groupby("run_time")["split"]
    .nunique()
)

assert leakage.max() == 1, "Data leakage detected between splits."

# Remove GSVO from test only

mask = (
    (df["split"] == "test")
    & (df["icao"] == "GSVO")
)

print(f"Removing {mask.sum():,} GSVO samples from test")

df = df.loc[~mask].copy()

# Save splits

sort_cols = ["run_time", "valid_time", "icao"]

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
        f" | stations={subset['icao'].nunique():2d}"
        f" | {subset['run_time'].min().date()} -> {subset['run_time'].max().date()}"
    )

print("\nSplit completed successfully.")