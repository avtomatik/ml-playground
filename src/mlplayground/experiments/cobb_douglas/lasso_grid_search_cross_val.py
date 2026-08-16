import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset


def run() -> None:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    lasso = Lasso()
    param_grid = {"alpha": np.logspace(-5, 0, 10)}
    gscv = GridSearchCV(
        estimator=lasso, param_grid=param_grid, cv=10, verbose=2
    )
    gscv.fit(X_train, y_train)
    model = gscv.best_estimator_
    score = model.score(X_test, y_test)
    # =============================================================================
    # usa_cobb_douglas0014.py
    #   TODO: Revise Fixed Assets Turnover Approximation with Lasso
    # =============================================================================
    print(f"Best parameters: {gscv.best_params_}")
    print(f"Test R²: {score:,.4f}")


if __name__ == "__main__":
    run()
