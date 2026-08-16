from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class DatasetProtocol(Protocol):
    """
    Contract for any dataset used by experiments.
    """

    name: str

    def load(self) -> pd.DataFrame: ...


@runtime_checkable
class EstimatorProtocol(Protocol):
    """
    Minimal sklearn-like estimator contract.
    """

    def fit(self, X, y): ...
    def predict(self, X) -> np.ndarray: ...


@runtime_checkable
class SupervisedDatasetProtocol(DatasetProtocol, Protocol):
    """
    Dataset with features and target.
    """

    feature_columns: tuple[str, ...]
    target_column: str

    def features(self) -> pd.DataFrame: ...
    def target(self) -> pd.Series: ...
