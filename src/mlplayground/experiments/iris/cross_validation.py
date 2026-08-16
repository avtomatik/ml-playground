import numpy as np
from sklearn.model_selection import (ShuffleSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mlplayground.datasets.iris import IrisDataset
from mlplayground.metrics.regression import print_cross_val_scores


def custom_cv_2folds(X: np.ndarray) -> tuple:
    """
    http://scikit-learn.org/stable/modules/cross_validation.html

    """
    n = X.shape[0]
    _ = 1
    while _ <= 2:
        idx = np.arange(n * (_ - 1) / 2, n * _ / 2, dtype=int)
        yield idx, idx
        _ += 1


def run() -> None:
    dataset = IrisDataset()
    X = dataset.features()
    y = dataset.target()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    # =========================================================================
    # Support Vector Machine: Support Vector Classification
    # =========================================================================
    estimator = SVC(kernel="linear", C=1).fit(X_train, y_train)

    scores = cross_val_score(estimator, X, y, cv=5)
    print_cross_val_scores(scores, text="Base Scoring")

    scores = cross_val_score(estimator, X, y, cv=5, scoring="f1_macro")
    print_cross_val_scores(scores, text="F1 Scoring")

    cross_validator = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(estimator, X, y, cv=cross_validator)
    print_cross_val_scores(scores, text="Shuffle Split")

    cross_validator = custom_cv_2folds(X)
    scores = cross_val_score(estimator, X, y, cv=cross_validator)
    print_cross_val_scores(scores, text="Custom")

    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    estimator = SVC(C=1).fit(X_train_transformed, y_train)
    X_test_transformed = scaler.transform(X_test)
    scores = estimator.score(X_test_transformed, y_test)
    print_cross_val_scores(scores, text="Standard Scaler")

    estimator = make_pipeline(StandardScaler(), SVC(C=1))
    cross_validator = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(estimator, X, y, cv=cross_validator)
    print_cross_val_scores(scores, text="Composite Estimator")


if __name__ == "__main__":
    run()
