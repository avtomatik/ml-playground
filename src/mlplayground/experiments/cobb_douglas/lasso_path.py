import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

from mlplayground.datasets.cobb_douglas import CobbDouglasDataset


def run() -> None:
    dataset = CobbDouglasDataset()
    X = dataset.features()
    y = dataset.target()
    alphas = np.logspace(-5, 2, 1000)
    _, coefs, _ = Lasso.path(X, y, alphas=alphas)
    fig, ax = plt.subplots()
    ax.plot(alphas, coefs.T)
    ax.set_xscale("log")
    ax.set_xlim(alphas.max(), alphas.min())
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Coefficient")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    run()
