import numpy as np

from src.models.base_model import BaseModel


class AromeBaseline(BaseModel):

    def __init__(
        self,
        gust_column="arome_gust60_speed",
        threshold=10.0,
    ):
        super().__init__(
            gust_column=gust_column,
            threshold=threshold,
        )

        self.gust_column = gust_column
        self.threshold = threshold

    def fit(self, X, y):
        # No training required
        return self

    def predict(self, X):

        if self.gust_column not in X.columns:
            raise ValueError(
                f"Column '{self.gust_column}' not found."
            )

        return (
            X[self.gust_column]
            >= self.threshold
        ).astype(int).to_numpy()

    def predict_proba(self, X):
        # Baseline has no probability output
        return None