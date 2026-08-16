import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def _as_1d(values) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected exactly one feature column.")
        return values.iloc[:, 0].to_numpy()
    return np.asarray(values).reshape(-1)


def plot_linear_and_quadratic_regression(
    X, y, X_fit, y_lin_fit, y_quad_fit
) -> None:
    X_values = _as_1d(X)
    y_values = _as_1d(y)
    X_fit_values = _as_1d(X_fit)
    plt.scatter(X_values, y_values, label="Trained")
    plt.plot(X_fit_values, y_lin_fit, label="Linear", linestyle="--")
    plt.plot(X_fit_values, y_quad_fit, label="Quadratic")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


def plot_predictions(X, y, y_pred, *, xlabel=None, ylabel=None) -> None:
    X_values = _as_1d(X)
    y_values = _as_1d(y)
    y_pred_values = _as_1d(y_pred)
    plt.scatter(X_values, y_values)
    plt.plot(X_values, y_pred_values)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.grid()
    plt.show()


def calculate_graph_k_folds_linear_regression(
    X: np.ndarray, y: np.ndarray, n_splits: int = 6
) -> None:
    model = LinearRegression().fit(X, y)
    # =========================================================================
    # K-Folds cross-validator
    # =========================================================================
    kf = KFold(n_splits=n_splits)

    y_container = np.zeros_like(y)

    for idx_train, idx_test in kf.split(X, y):
        model.fit(X.iloc[idx_train], y[idx_train])
        y_test_pred = model.predict(X.iloc[idx_test])
        y_container[idx_test] = y_test_pred
        y_pred = model.predict(X)

    plt.figure()
    plt.scatter(X, y, label="Original")
    plt.scatter(X, y_container, label="Linear Fit, K-Folds cross-validator")
    plt.scatter(X, y_pred, label="Linear Fit, Cumulative")
    plt.title(
        "`Labor Productivity` over `Labor Capital Intensity`, 1899--1922"
    )
    plt.xlabel("Labor Capital Intensity")
    plt.ylabel("Labor Productivity")
    plt.legend()
    plt.grid()
    plt.show()
