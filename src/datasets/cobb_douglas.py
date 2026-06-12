import pandas as pd

from core.paths import DATA_DIR


def load() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "gold" / "productivity_metrics.parquet")
