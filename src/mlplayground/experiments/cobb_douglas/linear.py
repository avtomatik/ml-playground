import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset
from mlplayground.experiments.registry import register
from mlplayground.visualization.regression import \
    plot_linear_and_quadratic_regression


@register("cobb-douglas-linear")
def run(*, visualize: bool = True) -> LinearRegression:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()
    linear_model = LinearRegression()
    quadratic_model = LinearRegression()
    quadratic_features = PolynomialFeatures(degree=2)
    X_quad = quadratic_features.fit_transform(X)
    linear_model.fit(X, y)
    quadratic_model.fit(X_quad, y)
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    y_lin_fit = linear_model.predict(X_fit)
    y_quad_fit = quadratic_model.predict(quadratic_features.transform(X_fit))
    if visualize:
        plot_linear_and_quadratic_regression(
            X, y, X_fit, y_lin_fit, y_quad_fit
        )
    return linear_model


if __name__ == "__main__":
    run()
