import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split

from datasets.cobb_douglas import load as load_cobb_douglas

if __name__ == "__main__":
    # =========================================================================
    # Make Dataset
    # =========================================================================
    df = load_cobb_douglas()

    X = df[["labor_capital_intensity"]]
    y = df["labor_productivity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1
    )

    ransac = RANSACRegressor(
        LinearRegression(),
        max_trials=100,
        min_samples=10,
        residual_threshold=0.05,
        random_state=0,
    )
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(X.min(), X.max(), 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])

    plt.scatter(
        X[inlier_mask], y[inlier_mask], c="blue", marker="o", label="Inliers"
    )
    plt.scatter(
        X[outlier_mask],
        y[outlier_mask],
        c="lightgreen",
        marker="s",
        label="Outliers",
    )
    plt.plot(line_X, line_y_ransac, color="red")
    plt.xlabel("Labor Capital Intensity")
    plt.ylabel("Labor Productivity")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
