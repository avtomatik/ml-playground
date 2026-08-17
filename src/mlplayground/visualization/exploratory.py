import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix


def plot_scatter_matrix(
    df: pd.DataFrame, title: str = "Scatter Matrix"
) -> None:
    """Plot a scatter matrix of numerical values in the DataFrame."""
    scatter_matrix(df, alpha=0.8, figsize=(8, 6), diagonal="hist")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
