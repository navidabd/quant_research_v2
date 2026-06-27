"""
Normalization utility.

Uses RobustScaler (median + IQR) instead of StandardScaler because
financial features always have fat tails and occasional extreme outliers.
StandardScaler would let one flash-crash spike distort the entire scale.

Usage at training time:

    from utils.normalize import Normalizer

    norm = Normalizer()
    X_train_scaled = norm.fit_transform(X_train, feature_cols)
    X_val_scaled   = norm.transform(X_val)

    norm.save("data/processed/scaler_BTC.pkl")

    # later, at inference time:
    norm2 = Normalizer.load("data/processed/scaler_BTC.pkl")
    X_new_scaled = norm2.transform(X_new)

Note: fit() only ever on training data. Never fit on validation or test data.
"""

import pickle

import numpy as np
import pandas as pd


class Normalizer:
    def __init__(self, quantile_range: tuple = (5, 95)):
        """
        quantile_range: IQR range for the scaler.
        Default (5, 95) is more aggressive than sklearn's (25, 75),
        which handles crypto's fat tails better.
        """
        self.quantile_range = quantile_range
        self._medians: pd.Series | None = None
        self._iqrs: pd.Series | None = None
        self._feature_cols: list[str] | None = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "Normalizer":
        self._feature_cols = feature_cols
        X = df[feature_cols].astype("float64")

        q_lo, q_hi = self.quantile_range
        self._medians = X.median()
        self._iqrs = X.quantile(q_hi / 100) - X.quantile(q_lo / 100)

        # if IQR is zero (constant column), set to 1 to avoid divide-by-zero
        self._iqrs = self._iqrs.replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._medians is None:
            raise RuntimeError("Normalizer has not been fit yet — call fit() first")

        X = df[self._feature_cols].astype("float64")
        scaled = (X - self._medians) / self._iqrs

        # winsorize to [-10, 10] so extreme outliers don't blow up linear models
        scaled = scaled.clip(-10, 10)
        return scaled.astype("float32")

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with open(path, "rb") as f:
            return pickle.load(f)