import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

from datasets.cobb_douglas import load as load_cobb_douglas


def main() -> None:
    """
    Elastic Net

    Returns
    -------
    None
        DESCRIPTION.

    """
    # =========================================================================
    # Make Dataset
    # =========================================================================
    df = load_cobb_douglas()

    X = df[["labor_capital_intensity"]]
    y = df["labor_productivity"]
    # =========================================================================
    # Process Dataset
    # =========================================================================
    solver = Lasso(normalize=1)
    alphas = np.logspace(-5, 2, 1000)
    alphas, coefs, _ = solver.path(X, y, alphas=alphas)
    # =========================================================================
    # Visualize
    # =========================================================================
    _fig, ax = plt.subplots()
    ax.plot(alphas, coefs.T)
    ax.set_xscale("log")
    ax.set_xlim(alphas.max(), alphas.min())
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
