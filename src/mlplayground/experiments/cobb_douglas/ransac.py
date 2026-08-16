import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset


def run() -> RANSACRegressor:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()
    X_values = X.iloc[:, 0].to_numpy()
    y_values = y.to_numpy()
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        max_trials=100,
        min_samples=10,
        residual_threshold=0.05,
        random_state=0,
    )
    ransac.fit(X_values[:, np.newaxis], y_values)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.linspace(
        X_values.min(),
        X_values.max(),
        100,
    )
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(
        X_values[inlier_mask],
        y_values[inlier_mask],
        marker="o",
        label="Inliers",
    )
    plt.scatter(
        X_values[outlier_mask],
        y_values[outlier_mask],
        marker="s",
        label="Outliers",
    )
    plt.plot(line_X, line_y_ransac)
    plt.xlabel("Labor Capital Intensity")
    plt.ylabel("Labor Productivity")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
    return ransac


if __name__ == "__main__":
    run()
