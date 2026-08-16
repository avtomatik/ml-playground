import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_boxplot(df: pd.DataFrame) -> None:
    """Show a boxplot of all numeric features."""
    df.boxplot()
    plt.title("Boxplot of Iris Features")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot a correlation heatmap of the dataset's numeric features."""
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="RdYlGn")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_sepal_scatter(df: pd.DataFrame) -> None:
    """Create a scatter plot of sepal width vs. sepal length."""
    color_map = {
        "setosa": "green",
        "versicolor": "red",
        "virginica": "blue",
    }
    colors = df["species"].map(color_map)
    df.plot.scatter(
        x="sepal_width",
        y="sepal_length",
        color=colors,
        title="Sepal Width vs Length by Species",
    )
    plt.tight_layout()
    plt.grid(True)
    plt.show()
