from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.metar_lag_features import (
    build_metar_history, ICAO_COL, RUN_TIME_COL, VALID_TIME_COL,
    WIND_COL, TEMP_COL, PRESSURE_COL,
)

MAX_LEN = 72  # e.g. last 72 hourly METAR obs before run_time
SEQ_COLS = [
    WIND_COL, TEMP_COL, PRESSURE_COL,
    "metar_wind_dir_sin", "metar_wind_dir_cos",
    "metar_hours_before_run",   # computed per-timestep, not in station_hist
]


def build_sequences_for_station(run_times, valid_times, values, max_len):
    run_times = run_times.astype("datetime64[ns]")
    end = np.searchsorted(valid_times, run_times, side="left")  # leakage-safe cutoff

    n, F = len(run_times), values.shape[1]
    F_out = F + 1  # +1 for hours_before_run
    seq = np.zeros((n, max_len, F_out), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=np.float32)

    for i, e in enumerate(end):
        start = max(0, e - max_len)
        window = values[start:e]
        L = len(window)
        if L == 0:
            continue

        # hours_before_run: strictly positive by construction (searchsorted
        # cutoff guarantees valid_time < run_time for every included obs)
        vt_window = valid_times[start:e]
        hours_before = (run_times[i] - vt_window) / np.timedelta64(1, "h")

        seq[i, -L:, :F] = window
        seq[i, -L:, F] = hours_before
        mask[i, -L:] = 1.0

    return seq, mask


def build_split_sequences(df: pd.DataFrame, hist: pd.DataFrame, max_len: int):
    pairs = df[[ICAO_COL, RUN_TIME_COL]].drop_duplicates().reset_index(drop=True)
    raw_cols = SEQ_COLS[:-1]  # exclude the derived hours_before_run column

    all_seq, all_mask = [], []
    for icao, g in pairs.groupby(ICAO_COL):
        station_hist = hist[hist[ICAO_COL] == icao]
        if station_hist.empty:
            n = len(g)
            all_seq.append(np.zeros((n, max_len, len(SEQ_COLS)), dtype=np.float32))
            all_mask.append(np.zeros((n, max_len), dtype=np.float32))
            continue

        seq, mask = build_sequences_for_station(
            g[RUN_TIME_COL].values,
            station_hist[VALID_TIME_COL].values.astype("datetime64[ns]"),
            station_hist[raw_cols].to_numpy(float),
            max_len,
        )
        all_seq.append(seq)
        all_mask.append(mask)

    seq = np.concatenate(all_seq, axis=0)
    mask = np.concatenate(all_mask, axis=0)

    ordered_pairs = pd.concat(
        [pairs[pairs[ICAO_COL] == icao] for icao, _ in pairs.groupby(ICAO_COL)],
        ignore_index=True,
    )
    pair_id = (ordered_pairs[ICAO_COL].astype(str) + "_" +
               ordered_pairs[RUN_TIME_COL].astype(str)).to_numpy(dtype="U")

    return seq, mask, pair_id


def main():
    full_df = pd.read_csv(
        PROJECT_ROOT / "data" / "cleaned" / "FINAL_PREPARED_DATA_2021_2025__gust_excess.csv",
        parse_dates=[RUN_TIME_COL, VALID_TIME_COL],
    )
    hist = build_metar_history(full_df)

    splits_dir = PROJECT_ROOT / "data" / "processed" / "splits_gust_excess"
    out_dir = splits_dir  # save alongside the parquets

    for name in ["train", "val", "test"]:
        df = pd.read_parquet(splits_dir / f"{name}.parquet")
        seq, mask, pair_id = build_split_sequences(df, hist, MAX_LEN)

        if name == "train":
            flat = seq.reshape(-1, seq.shape[-1]).astype(np.float32)
            valid = mask.reshape(-1).astype(bool)

            scaler = StandardScaler()
            scaler.fit(flat[valid])

            joblib.dump(scaler, out_dir / "history_scaler.joblib")

        np.savez(
            out_dir / f"{name}_history.npz",
            seq=seq, mask=mask, pair_id=pair_id,
        )
        print(f"{name}: {seq.shape[0]:,} unique (icao, run_time) pairs -> {seq.shape} "
              f"(F={len(SEQ_COLS)}: {SEQ_COLS})")


if __name__ == "__main__":
    main()