import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

from datasets.cobb_douglas import load as load_cobb_douglas

if __name__ == "__main__":
    # =========================================================================
    # Make Dataset
    # =========================================================================
    df = load_cobb_douglas()

    X = df[["labor_capital_intensity"]]
    y = df["labor_productivity"]

    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)

    lr.fit(X, y)
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)

    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

    plt.scatter(X, y, label="Trained")
    plt.plot(X_fit, y_lin_fit, label="Linear", linestyle="--")
    plt.plot(X_fit, y_quad_fit, label="Quadratic")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


def calculate_graph_k_folds_linear_regression(
    X: np.ndarray, y: np.ndarray, n_splits: int = 6
) -> None:
    solver = LinearRegression().fit(X, y)
    # =========================================================================
    # K-Folds cross-validator
    # =========================================================================
    kf = KFold(n_splits=n_splits)

    y_container = np.zeros_like(y)

    for idx_train, idx_test in kf.split(X, y):
        solver.fit(X[idx_train], y[idx_train])
        y_test_pred = solver.predict(X[idx_test])
        y_container[idx_test] = y_test_pred
        y_pred = solver.predict(X)

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
    print("Figure Has Been Plotted")


def compare_r2s_print_out_coefs(X: np.ndarray, y: np.ndarray) -> None:
    # =========================================================================
    # TODO: Split Function to Increase Cohesion
    # =========================================================================
    solver = LinearRegression().fit(X, y)
    y_pred = solver.predict(X)

    r2_solver = solver.score(X, y)
    r2_metrics = r2_score(y, y_pred)

    print(solver.coef_)
    print(solver.intercept_)
    print(
        f"R**2 Powered by sklearn.linear_model.LinearRegression: {r2_solver:.6}"
    )
    print(f"R**2 Powered by sklearn.metrics.r2_score: {r2_metrics:.6}")


def get_neg_mean_squared_error_leave_one_out(
    X: np.ndarray, y: np.ndarray
) -> None:
    """
    Cross Validation

    Returns
    -------
    None.

    """
    solver = LinearRegression().fit(X, y)

    loo = LeaveOneOut()

    scores = cross_val_score(
        solver, X, y, scoring="neg_mean_squared_error", cv=loo
    )
    print(f"Mean of `neg_mean_squared_error`: {scores.mean():,.6f}")
