from pathlib import Path
import sys


import numpy as np
import pandas as pd

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025_with_rolling_features.csv"

ICAO_COL = "icao"
RUN_TIME_COL = "run_time"
VALID_TIME_COL = "valid_time"

WIND_COL = "metar_wind_speed_ms"
TEMP_COL = "metar_temp_c"
PRESSURE_COL = "metar_slp_hpa"

INCLUDE_STALENESS_FEATURE = True

FEATURE_COLUMNS = [
    "metar_wind_mean_3h", "metar_wind_mean_6h",
    "metar_wind_std_3h", "metar_wind_std_6h",
    "metar_wind_max_3h", "metar_wind_max_6h",
    "metar_temp_mean_3h", "metar_temp_mean_6h",
    "metar_pressure_mean_6h", "metar_pressure_std_6h",
    "metar_wind_trend_6h", "metar_temp_trend_6h", "metar_pressure_trend_6h",
]


# Helpers

def _safe_mean(arr: np.ndarray) -> float:
    return np.nan if arr.size == 0 else float(np.nanmean(arr))


def _safe_max(arr: np.ndarray) -> float:
    return np.nan if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmax(arr))


def _safe_std(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)] if arr.size else arr
    if valid.size < 2:
        return np.nan
    return float(np.std(valid, ddof=1))


# Step 1 — unique METAR observation history per station

def build_metar_history(df: pd.DataFrame) -> pd.DataFrame:
    cols = [ICAO_COL, VALID_TIME_COL, WIND_COL, TEMP_COL, PRESSURE_COL]
    return (
        df[cols]
        .dropna(subset=[VALID_TIME_COL])
        .groupby([ICAO_COL, VALID_TIME_COL], as_index=False)
        .agg({c: "first" for c in (WIND_COL, TEMP_COL, PRESSURE_COL)})
        .sort_values([ICAO_COL, VALID_TIME_COL])
        .reset_index(drop=True)
    )


# Step 2 — leakage-safe windowed features via searchsorted, per station

def compute_features_for_station(
    run_times: np.ndarray,
    valid_times: np.ndarray,
    wind: np.ndarray,
    temp: np.ndarray,
    pressure: np.ndarray,
) -> pd.DataFrame:
    run_times = run_times.astype("datetime64[ns]")
    t3 = run_times - np.timedelta64(3, "h")
    t6 = run_times - np.timedelta64(6, "h")


    end = np.searchsorted(valid_times, run_times, side="left")
    s3 = np.searchsorted(valid_times, t3, side="left")
    s6 = np.searchsorted(valid_times, t6, side="left")

    out = {c: np.empty(len(run_times)) for c in FEATURE_COLUMNS}
    hours_since_last_obs = np.empty(len(run_times))

    for i in range(len(run_times)):
        a, b, e = s3[i], s6[i], end[i]

        w3, w6, wp = wind[a:e], wind[b:e], wind[b:a]
        t3a, t6a, tp = temp[a:e], temp[b:e], temp[b:a]
        p3, p6, pp = pressure[a:e], pressure[b:e], pressure[b:a]

        wm3, wm6, wmp = _safe_mean(w3), _safe_mean(w6), _safe_mean(wp)
        tm3, tm6, tmp = _safe_mean(t3a), _safe_mean(t6a), _safe_mean(tp)
        pm3, pm6, pmp = _safe_mean(p3), _safe_mean(p6), _safe_mean(pp)

        out["metar_wind_mean_3h"][i] = wm3
        out["metar_wind_mean_6h"][i] = wm6
        out["metar_wind_std_3h"][i] = _safe_std(w3)
        out["metar_wind_std_6h"][i] = _safe_std(w6)
        out["metar_wind_max_3h"][i] = _safe_max(w3)
        out["metar_wind_max_6h"][i] = _safe_max(w6)

        out["metar_temp_mean_3h"][i] = tm3
        out["metar_temp_mean_6h"][i] = tm6

        out["metar_pressure_mean_6h"][i] = pm6
        out["metar_pressure_std_6h"][i] = _safe_std(p6)

        # Trend = mean(last 3h) - mean(previous 3h), smoother than a single point-in-time difference.
        out["metar_wind_trend_6h"][i] = wm3 - wmp
        out["metar_temp_trend_6h"][i] = tm3 - tmp
        out["metar_pressure_trend_6h"][i] = pm3 - pmp

        if e > 0:
            hours_since_last_obs[i] = (run_times[i] - valid_times[e - 1]) / np.timedelta64(1, "h")
        else:
            hours_since_last_obs[i] = np.nan

    res = pd.DataFrame(out)
    res.insert(0, RUN_TIME_COL, run_times)
    if INCLUDE_STALENESS_FEATURE:
        res["metar_hours_since_last_obs"] = hours_since_last_obs
    return res


# Step 3 — apply per station, over unique (icao, run_time) pairs only

def compute_all_rolling_features(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    pairs = df[[ICAO_COL, RUN_TIME_COL]].drop_duplicates()

    all_feature_cols = FEATURE_COLUMNS + (
        ["metar_hours_since_last_obs"] if INCLUDE_STALENESS_FEATURE else []
    )

    results = []
    for icao, g in pairs.groupby(ICAO_COL):
        station_hist = hist[hist[ICAO_COL] == icao]

        if station_hist.empty:
            f = pd.DataFrame({RUN_TIME_COL: g[RUN_TIME_COL].values})
            for c in all_feature_cols:
                f[c] = np.nan
        else:
            f = compute_features_for_station(
                g[RUN_TIME_COL].values,
                station_hist[VALID_TIME_COL].values.astype("datetime64[ns]"),
                station_hist[WIND_COL].to_numpy(float),
                station_hist[TEMP_COL].to_numpy(float),
                station_hist[PRESSURE_COL].to_numpy(float),
            )

        f.insert(0, ICAO_COL, icao)
        results.append(f)

    return pd.concat(results, ignore_index=True)


# Main

def main(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path, parse_dates=[RUN_TIME_COL, VALID_TIME_COL])
    n_rows = len(df)
    n_pairs = df[[ICAO_COL, RUN_TIME_COL]].drop_duplicates().shape[0]
    print(f"  {n_rows:,} rows | {n_pairs:,} unique (icao, run_time) pairs "
          f"-> {n_rows / n_pairs:.1f}x reduction in rolling-feature computation")

    print("Building unique METAR observation history ...")
    hist = build_metar_history(df)
    print(f"  {len(hist):,} unique METAR observations")

    print("Computing 3h / 6h rolling + trend features per (icao, run_time) ...")
    feats = compute_all_rolling_features(df, hist)

    all_feature_cols = FEATURE_COLUMNS + (
        ["metar_hours_since_last_obs"] if INCLUDE_STALENESS_FEATURE else []
    )

    print("Merging back onto full dataset ...")
    out = df.merge(feats, on=[ICAO_COL, RUN_TIME_COL], how="left", validate="many_to_one")
    assert len(out) == n_rows, "Row count changed during merge — check for duplicate keys."

    # Leakage guard: for every row where we found at least one prior METAR
    # observation, the implied "last observation time" must be strictly
    # before run_time. We recover it from metar_hours_since_last_obs (>0)
    # if present; otherwise this check is skipped.
    if INCLUDE_STALENESS_FEATURE:
        matched = out["metar_hours_since_last_obs"].notna()
        assert (out.loc[matched, "metar_hours_since_last_obs"] > 0).all(), (
            "Leakage detected: a METAR observation at/after run_time was used."
        )

    print("\nFeature coverage (non-null %):")
    for c, p in (out[all_feature_cols].notna().mean() * 100).items():
        print(f"  {c:<32}{p:5.1f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}  ({out.shape[0]:,} rows, {out.shape[1]} cols)")

    return out


if __name__ == "__main__":
    main()