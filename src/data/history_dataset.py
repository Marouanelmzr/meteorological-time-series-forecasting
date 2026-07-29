from pathlib import Path

from hydra.utils import to_absolute_path

import numpy as np
import pandas as pd


class HistoryDataset:

    ICAO_COL = "icao"
    RUN_TIME_COL = "run_time"

    def __init__(self, data_dir: Path, seq_features: list[str], max_len: int = 72):
        self.data_dir = Path(to_absolute_path(str(data_dir)))
        self.seq_features = seq_features
        self.max_len = max_len

    @staticmethod
    def _pair_id(df: pd.DataFrame) -> np.ndarray:
        return (
            df[HistoryDataset.ICAO_COL].astype(str)
            + "_"
            + df[HistoryDataset.RUN_TIME_COL].astype(str)
        ).to_numpy()

    def _load(self, filename: str, df: pd.DataFrame):
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        arr = np.load(path)
        seq, mask, pair_id = arr["seq"], arr["mask"], arr["pair_id"]

        lookup = {pid: i for i, pid in enumerate(pair_id)}
        row_pair_id = self._pair_id(df)

        missing = [pid for pid in row_pair_id if pid not in lookup]
        if missing:
            raise ValueError(
                f"{filename}: {len(missing)} row(s) with no matching precomputed "
                f"sequence — rebuild build_history_sequences.py against the current "
                f"split. First missing key: {missing[0]}"
            )

        idx = np.array([lookup[pid] for pid in row_pair_id])
        return seq[idx].astype(np.float32), mask[idx].astype(np.float32)

    def train(self, dataset: "Dataset"):
        return self._load("train_history.npz", dataset.train_df)

    def validation(self, dataset: "Dataset"):
        return self._load("val_history.npz", dataset.val_df)

    def test(self, dataset: "Dataset"):
        return self._load("test_history.npz", dataset.test_df)