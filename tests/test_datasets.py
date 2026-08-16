import pandas as pd

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset
from mlplayground.datasets.iris import IrisDataset


def test_cobb_douglas_dataset():
    dataset = CobbDouglasDataset()
    frame = dataset.load()
    features = dataset.features()
    target = dataset.target()
    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert list(features.columns) == list(dataset.feature_columns)
    assert features.shape[0] == target.shape[0]
    assert dataset.target_column in frame.columns


def test_iris_dataset():
    dataset = IrisDataset()
    frame = dataset.load()
    features = dataset.features()
    target = dataset.target()
    assert not frame.empty
    assert list(features.columns) == list(dataset.feature_columns)
    assert target.name == dataset.target_column
    assert len(features) == len(target)
    assert "species" in frame.columns
