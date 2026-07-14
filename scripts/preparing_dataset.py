"""
Prepare merged AROME + METAR dataset for machine learning.

Pipeline
--------
1. Remove duplicate columns from the merge.
2. Rename columns.
3. Create temporal features.
4. Compute wind speed and direction.
5. Clean gust labels (<10 m/s -> non-gust).
6. Keep only training columns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(PROJECT_ROOT))

SRC = PROJECT_ROOT / "data" / "cleaned" / "AROME_METAR_merged_2021_2025.csv"
OUT = PROJECT_ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025.csv"

GUST_THRESHOLD = 10.0


def clean_columns(df):

    df.drop(
        columns=["id", "station", "date"],
        errors="ignore",
        inplace=True,
    )

    df.rename(
        columns={
            "wmo_id": "station_id",
            "lon": "arome_lon",
            "lat": "arome_lat",
            "longitude": "station_lon",
            "latitude": "station_lat",
            "datetime": "valid_time",
            "name": "station_name",
            "source": "metar_source",
            "wind_dir_deg": "metar_wind_dir_deg",
            "wind_speed_ms": "metar_wind_speed_ms",
            "visibility_m": "metar_visibility_m",
            "temp_c": "metar_temp_c",
            "dewpoint_c": "metar_dewpoint_c",
            "slp_hpa": "metar_slp_hpa",
            "ceiling_m": "metar_ceiling_m",
            "clouds": "metar_clouds",
            "present_weather": "metar_present_weather",
            "avwx_gust_ms": "metar_avwx_gust_ms",
            "gust_mismatch": "metar_gust_mismatch",
        },
        inplace=True,
    )

    return df


def add_time_features(df):

    # All times are in UTC, so we can use the valid_time to compute the run_time and lead_time.

    run_time = df["valid_time"].dt.normalize()

    midnight = df["valid_time"].dt.hour == 0
    run_time.loc[midnight] -= pd.Timedelta(days=1)

    df["run_time"] = run_time

    df["lead_time"] = (
        (df["valid_time"] - df["run_time"])
        .dt.total_seconds()
        .div(3600)
        .astype(int)
    )

    doy = df["valid_time"].dt.dayofyear

    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    hour = df["valid_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return df


def add_wind_features(df):

    df["t2m"] = df["t2m"] - 273.15
    df["psurf"] = df["psurf"] / 100

    for name, u, v in [
        ("wind10", "u10", "v10"),
        ("wind850", "u850", "v850"),
        ("wind950", "u950", "v950"),
        ("gust60", "u_gust60", "v_gust60"),
    ]:

        df[f"arome_{name}_speed"] = np.hypot(df[u], df[v])

        df[f"arome_{name}_dir"] = (
            np.degrees(np.arctan2(-df[u], -df[v])) % 360
        )
    
    df["wind_shear_950_10"] = np.hypot(
    df["u950"] - df["u10"],
    df["v950"] - df["v10"],
)

    return df


def clean_targets(df):

    mask = (
        (df["has_gust"] == 1)
        & (df["gust_speed_ms"] < GUST_THRESHOLD)
    )

    print(f"Reclassified gusts : {mask.sum()}")

    df["gust_reclassified_noise"] = mask

    df.loc[mask, "has_gust"] = 0
    df.loc[mask, "gust_speed_ms"] = np.nan

    return df


KEEP_COLUMNS = [

    # metadata
    "station_id","icao","station_name",
    "arome_lon","arome_lat",
    "station_lon","station_lat",
    "elevation_m","distance_km",

    # time
    "run_time","valid_time","lead_time",
    "doy_sin","doy_cos",
    "hour_sin","hour_cos",

    # AROME
    "u10","v10","arome_wind10_speed","arome_wind10_dir",
    "u850","v850","arome_wind850_speed","arome_wind850_dir",
    "u950","v950","arome_wind950_speed","arome_wind950_dir",
    "t2m","rh2m","psurf","pblh","tke20m","edr20m",
    "u_gust60","v_gust60",
    "arome_gust60_speed","arome_gust60_dir",
    "wind_shear_950_10",

    # METAR
    "metar_source",
    "metar_wind_dir_deg",
    "metar_wind_speed_ms",
    "metar_visibility_m",
    "metar_temp_c",
    "metar_dewpoint_c",
    "metar_slp_hpa",
    "metar_ceiling_m",
    "metar_clouds",
    "metar_present_weather",
    "metar_avwx_gust_ms",
    "metar_gust_mismatch",

    # targets
    "has_gust",
    "gust_speed_ms",
    "gust_reclassified_noise",
]


def main():

    df = pd.read_csv(
        SRC,
        parse_dates=["datetime"],
    )

    df = clean_columns(df)
    df = add_time_features(df)
    df = add_wind_features(df)
    df = clean_targets(df)

    df = df[KEEP_COLUMNS]

    df.to_csv(OUT, index=False)

    print(df.shape)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()