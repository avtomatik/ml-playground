from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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

    model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    plot_residuals(y, y_train, y_test, y_train_pred, y_test_pred)
    evaluate_regression(y_train, y_test, y_train_pred, y_test_pred)


if __name__ == "__main__":
    run()
