from sklearn.tree import DecisionTreeRegressor

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset
from mlplayground.visualization.regression import plot_predictions


def run() -> DecisionTreeRegressor:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()
    tree = DecisionTreeRegressor(max_depth=1, random_state=0)
    tree.fit(X, y)
    sort_idx = X.iloc[:, 0].to_numpy().argsort()
    X_sorted = X.iloc[sort_idx]
    y_sorted = y.iloc[sort_idx]
    y_pred_sorted = tree.predict(X_sorted)
    plot_predictions(
        X_sorted,
        y_sorted,
        y_pred_sorted,
        xlabel="Labor Capital Intensity",
        ylabel="Labor Productivity",
    )
    return tree


if __name__ == "__main__":
    run()
