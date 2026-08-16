import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score


def summarize_regression_model(
    X: np.ndarray, y: np.ndarray, model: RegressorMixin
) -> None:
    model.fit(X, y)
    y_pred = model.predict(X)
    r2_model = model.score(X, y)
    r2_metric = r2_score(y, y_pred)
    print(model.coef_)
    print(model.intercept_)
    print(f"R² from {model.__class__.__name__}.score: {r2_model:.6}")
    print(f"R² from sklearn.metrics.r2_score: {r2_metric:.6}")


def get_neg_mean_squared_error_leave_one_out(
    X: np.ndarray, y: np.ndarray
) -> None:
    model = LinearRegression().fit(X, y)

    loo = LeaveOneOut()

    scores = cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=loo
    )
    print(f"Mean of `neg_mean_squared_error`: {scores.mean():,.6f}")


def evaluate_regression(y_train, y_test, y_train_pred, y_test_pred):
    print(
        f"MSE on Train Data: {mean_squared_error(y_train, y_train_pred):,.4f}"
    )
    print(f"MSE on Test Data: {mean_squared_error(y_test, y_test_pred):,.4f}")
    print(f"R² on Train Data: {r2_score(y_train, y_train_pred):,.4f}")
    print(f"R² on Test Data: {r2_score(y_test, y_test_pred):,.4f}")


def print_cross_val_scores(scores, text: str) -> None:
    print(f"Cross Validation: {text}")
    print(f"Accuracy: {scores.mean():,.4f} (+/- {2 * scores.std():,.4f})")
