from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025__no_t_with_rolling_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025__gust_excess.csv"


def main():

    print(f"Loading {INPUT_PATH}...")

    df = pd.read_csv(INPUT_PATH)

    print(f"{len(df):,} rows loaded")

    # Fill missing values
    df["gust_speed_ms"] = df["gust_speed_ms"].fillna(0.0)
    df["arome_gust60_speed"] = df["arome_gust60_speed"].fillna(0.0)

    # Compute gust excess 
    wind = df["metar_wind_speed_ms"].fillna(0.0)
    gust = df["gust_speed_ms"]

    df["gust_excess_ms"] = np.maximum(gust - wind, 0.0)

    print("\nGust excess statistics")
    print("----------------------")
    print(df["gust_excess_ms"].describe())

    print(f"\nZero targets : {(df['gust_excess_ms'] == 0).mean()*100:.2f}%")
    print(f"Positive     : {(df['gust_excess_ms'] > 0).mean()*100:.2f}%")

    # New regression target
    # target = gust_speed_ms - arome_gust60_speed

    df["gust_forecast_error_ms"] = df["gust_speed_ms"] - df["arome_gust60_speed"]

    print("\nRegression target statistics")
    print("----------------------------")
    print(df["gust_forecast_error_ms"].describe())

    print(f"\nMissing targets : {df['gust_forecast_error_ms'].isna().sum():,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()