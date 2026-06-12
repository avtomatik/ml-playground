import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from datasets.cobb_douglas import load as load_cobb_douglas


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="blue")
    plt.plot(X, model.predict(X), color="red")
    return


if __name__ == "__main__":
    # =========================================================================
    # Make Dataset
    # =========================================================================
    df = load_cobb_douglas()

    X = df[["labor_capital_intensity"]]
    y = df["labor_productivity"]

    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(X, y)
    sort_idx = X.flatten().argsort()

    lin_regplot(X[sort_idx], y[sort_idx], tree)
    plt.xlabel("Labor Capital Intensity")
    plt.ylabel("Labor Productivity")
    plt.grid()
    plt.show()
