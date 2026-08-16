from mlplayground.core.protocols import (DatasetProtocol, EstimatorProtocol,
                                         SupervisedDatasetProtocol)


def test_protocols_are_defined():
    assert DatasetProtocol is not None
    assert EstimatorProtocol is not None
    assert SupervisedDatasetProtocol is not None
