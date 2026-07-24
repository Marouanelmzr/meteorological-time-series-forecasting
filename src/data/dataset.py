from pathlib import Path

from hydra.utils import to_absolute_path

import pandas as pd


class Dataset:

    def __init__(
        self,
        data_dir: Path,
        target: str,
        feature_columns: list[str],
        metadata_columns: list[str] | None = None,
    ):
        self.data_dir = Path(
            to_absolute_path(str(data_dir))
        )

        self.target = target
        self.feature_columns = feature_columns
        self.metadata_columns = metadata_columns or []

        self.train_df = self._load("train.parquet")
        self.val_df = self._load("val.parquet")
        self.test_df = self._load("test.parquet")

    def _load(self, filename: str) -> pd.DataFrame:

        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(path)

        return pd.read_parquet(path)

    def _split(self, df: pd.DataFrame):

        X = df[self.feature_columns].copy()

        y = df[self.target].copy()

        metadata = (
            df[self.metadata_columns].copy()
            if self.metadata_columns
            else pd.DataFrame(index=df.index)
        )

        return X, y, metadata

    def _split_positive(self, df: pd.DataFrame):

        mask = df[self.target] > 0

        X = df.loc[mask, self.feature_columns].copy()

        y = df.loc[mask, self.target].copy()

        metadata = (
            df.loc[mask, self.metadata_columns].copy()
            if self.metadata_columns
            else pd.DataFrame(index=df.loc[mask].index)
        )

        return X, y, metadata

    def train(self):

        return self._split(self.train_df)

    def validation(self):

        return self._split(self.val_df)

    def test(self):

        return self._split(self.test_df)

    def train_positive(self):

        return self._split_positive(self.train_df)

    def validation_positive(self):

        return self._split_positive(self.val_df)

    def test_positive(self):

        return self._split_positive(self.test_df)

    @property
    def feature_names(self):

        return self.feature_columns

    @property
    def target_name(self):

        return self.target

    @property
    def n_features(self):

        return len(self.feature_columns)

    @property
    def n_train(self):

        return len(self.train_df)

    @property
    def n_validation(self):

        return len(self.val_df)

    @property
    def n_test(self):

        return len(self.test_df)

    def summary(self):

        print("=" * 60)
        print("Dataset summary")
        print("=" * 60)
        
        print(f"Train rows          : {self.n_train:,}")
        print(f"Validation rows     : {self.n_validation:,}")
        print(f"Test rows           : {self.n_test:,}")
        
        print()
        
        print(
            f"Positive train      : {(self.train_df[self.target] > 0).sum():,}"
        )
        print(
            f"Positive validation : {(self.val_df[self.target] > 0).sum():,}"
        )
        print(
            f"Positive test       : {(self.test_df[self.target] > 0).sum():,}"
        )
        
        print()
        
        print(f"Target      : {self.target}")
        print(f"Features    : {len(self.feature_columns)}")