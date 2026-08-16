from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from mlplayground.core.paths import DATA_DIR


@dataclass(frozen=True)
class CobbDouglasDataset:
    name: str = "cobb-douglas"
    feature_columns: tuple[str, ...] = ("labor_capital_intensity",)
    target_column: str = "labor_productivity"

    def load(self) -> pd.DataFrame:
        return self.dataframe()

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        # TODO
        return pd.read_parquet(DATA_DIR / "bronze" / "cobb-douglas.parquet")

    def features(self) -> pd.DataFrame:
        return self.dataframe[list(self.feature_columns)]

    def target(self) -> pd.Series:
        return self.dataframe[self.target_column]
