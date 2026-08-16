from dataclasses import dataclass
from functools import cached_property

import pandas as pd
from sklearn.datasets import load_iris


@dataclass(frozen=True)
class IrisDataset:
    name: str = "iris"
    feature_columns: tuple[str, ...] = (
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    )
    target_column: str = "target"

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df.columns = [*self.feature_columns, self.target_column]
        df["species"] = pd.Categorical.from_codes(
            iris.target,
            iris.target_names,
        )
        return df

    def load(self) -> pd.DataFrame:
        return self.dataframe

    def features(self) -> pd.DataFrame:
        return self.dataframe[list(self.feature_columns)]

    def target(self) -> pd.Series:
        return self.dataframe[self.target_column]
