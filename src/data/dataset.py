from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "processed" / "train.parquet"
TEST_FILE = ROOT / "data" / "processed" / "test.parquet"
VAL_FILE = ROOT / "data" / "processed" / "val.parquet"

class Dataset:
    def __init__(
            self,
            data_dir: Path,
            target: str,
            features: list[str],
    ):
        self.data_dir = Path(data_dir)
        self.target = target
        self.features = features
        self.train_df = self._load("train.parquet")
        self.test_df = self._load("test.parquet")
        self.val_df = self._load("val.parquet")
    
    def _load(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_parquet(path)
    
    def _split(self, df: pd.DataFrame):
        X = df[self.features].copy()
        y = df[self.target].copy()
        return X, y
    
    def train(self):
        return self._split(self.train_df)
    
    def validation(self):
        return self._split(self.val_df)
    
    def test(self):
        return self._split(self.test_df)
    
    @property
    def n_features(self):
        return len(self.features)
    
    @property
    def feature_names(self):
        return self.features
    
    @property
    def target_name(self):
        return self.target
