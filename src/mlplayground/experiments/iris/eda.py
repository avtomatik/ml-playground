#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:49:26 2022

@author: alexandermikhailov

Exploratory Data Analysis on the Iris Dataset.

Visualizations:
- Boxplot of features
- Correlation heatmap
- Sepal scatter plot colored by species
"""
from mlplayground.datasets.iris import load_dataframe
from mlplayground.visualization.classification import (
    plot_boxplot, plot_correlation_heatmap, plot_sepal_scatter)


def run() -> None:
    # =========================================================================
    # Brett Vanderblock
    # =========================================================================
    df_iris = load_dataframe()
    plot_boxplot(df_iris)
    plot_correlation_heatmap(df_iris)
    plot_sepal_scatter(df_iris)


if __name__ == "__main__":
    run()
