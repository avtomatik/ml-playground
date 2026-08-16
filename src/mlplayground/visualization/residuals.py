import numpy as np
from matplotlib import pyplot as plt


def plot_residuals(y, y_train, y_test, y_train_pred, y_test_pred) -> None:
    y_train_pred = np.asarray(y_train_pred)
    y_test_pred = np.asarray(y_test_pred)
    train_residuals = y_train_pred - np.asarray(y_train)
    test_residuals = y_test_pred - np.asarray(y_test)
    all_predictions = np.concatenate([y_train_pred, y_test_pred])
    plt.scatter(
        y_train_pred,
        train_residuals,
        marker="o",
        s=35,
        alpha=0.5,
        label="Train",
    )
    plt.scatter(
        y_test_pred,
        test_residuals,
        marker="s",
        s=35,
        alpha=0.7,
        label="Test",
    )
    x_min = all_predictions.min()
    x_max = all_predictions.max()
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.legend(loc="upper right")
    plt.hlines(y=0, xmin=x_min, xmax=x_max, lw=2)
    plt.xlim([x_min, x_max])
    plt.grid()
    plt.show()
