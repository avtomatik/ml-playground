from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset
from mlplayground.metrics.regression import evaluate_regression
from mlplayground.visualization.residuals import plot_residuals


def run() -> None:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.45, random_state=1
    )

    forest = RandomForestRegressor(
        n_estimators=1000, criterion="squared_error", random_state=1, n_jobs=-1
    )
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    plot_residuals(y, y_train, y_test, y_train_pred, y_test_pred)
    evaluate_regression(y_train, y_test, y_train_pred, y_test_pred)


if __name__ == "__main__":
    run()
