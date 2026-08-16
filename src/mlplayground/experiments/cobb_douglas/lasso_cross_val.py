from sklearn.linear_model import LassoCV

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset
from mlplayground.metrics.regression import summarize_regression_model


def run() -> None:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()
    model = LassoCV(cv=4, random_state=0)
    summarize_regression_model(X, y, model)


if __name__ == "__main__":
    run()
