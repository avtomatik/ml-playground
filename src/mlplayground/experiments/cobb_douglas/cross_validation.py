import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (KFold, LeaveOneOut, LeavePOut,
                                     RepeatedKFold, ShuffleSplit,
                                     TimeSeriesSplit)

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset


def _plot_folds(X, y, splitter) -> None:
    plt.figure()
    plt.scatter(X, y)
    for split_number, (idx_train, _) in enumerate(splitter.split(X), start=1):
        x_train = X.iloc[idx_train].to_numpy().reshape(-1)
        y_train = y.iloc[idx_train].to_numpy()
        polyfit_linear = np.polyfit(x_train, y_train, deg=1)
        y_train_pred = np.poly1d(polyfit_linear)(x_train)
        plt.plot(
            x_train,
            y_train_pred,
            label=f"Split {split_number:02d}",
        )
    plt.legend()
    plt.grid()


def run() -> None:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()
    _plot_folds(X, y, KFold(n_splits=4))
    _plot_folds(
        X, y, RepeatedKFold(n_splits=2, n_repeats=2, random_state=12883823)
    )
    _plot_folds(X, y, LeaveOneOut())
    _plot_folds(X, y, LeavePOut(p=2))
    _plot_folds(X, y, ShuffleSplit(n_splits=2, test_size=0.25, random_state=0))
    _plot_folds(X, y, TimeSeriesSplit(n_splits=3))
    x_values = X.to_numpy().reshape(-1)
    y_values = y.to_numpy()
    polyfit_linear = np.polyfit(x_values, y_values, deg=1)
    y_pred = np.poly1d(polyfit_linear)(x_values)
    plt.plot(x_values, y_pred, label="Full dataset")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run()
