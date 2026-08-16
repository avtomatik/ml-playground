import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset


def run() -> None:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()

    model = LinearRegression()

    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # =========================================================================
    # Linear Fit
    # =========================================================================
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    model = model.fit(X, y)
    y_lin_fit = model.predict(X_fit)
    linear_r2 = r2_score(y, model.predict(X))

    # =========================================================================
    # Quadratic Fit
    # =========================================================================
    model = model.fit(X_quad, y)
    y_quad_fit = model.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, model.predict(X_quad))

    # =========================================================================
    # Cubic Fit
    # =========================================================================
    model = model.fit(X_cubic, y)
    y_cubic_fit = model.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, model.predict(X_cubic))

    # =========================================================================
    # Plot the Results
    # =========================================================================
    plt.scatter(X, y, label="Train", color="lightgray")
    plt.plot(
        X_fit,
        y_lin_fit,
        label=f"Linear (d=1), $R^2={linear_r2:,.4f}$",
        color="blue",
        lw=2,
        linestyle=":",
    )
    plt.plot(
        X_fit,
        y_quad_fit,
        label=f"Linear (d=2), $R^2={quadratic_r2:,.4f}$",
        color="red",
        lw=2,
        linestyle="-",
    )
    plt.plot(
        X_fit,
        y_cubic_fit,
        label=f"Linear (d=2), $R^2={cubic_r2:,.4f}$",
        color="green",
        lw=2,
        linestyle="--",
    )
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run()
