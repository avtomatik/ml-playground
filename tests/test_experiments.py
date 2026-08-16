import numpy as np
from sklearn.linear_model import LinearRegression

from mlplayground.experiments.cobb_douglas.linear import run
from mlplayground.experiments.registry import EXPERIMENTS, load_experiments


def test_linear_experiment():
    model = run(visualize=False)
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "predict")
    assert model.coef_.shape == (1,)
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.intercept_)


def test_experiment_registry():
    EXPERIMENTS.clear()
    load_experiments()
    assert "cobb-douglas-linear" in EXPERIMENTS
    assert callable(EXPERIMENTS["cobb-douglas-linear"])
