# ML Playground

A lightweight machine learning sandbox for experimenting with datasets, models, validation strategies, metrics, and visualizations.

The project is built around reusable dataset abstractions and isolated experiments, making it easy to compare different machine learning approaches while keeping data loading, evaluation, and visualization organized.

## Features

- Dataset abstractions with reusable interfaces
- Regression and classification experiments
- Model comparison using scikit-learn estimators
- Cross-validation experiments
- Regression metrics and residual analysis
- Exploratory data analysis utilities
- Visualization helpers
- Experiment registry for discoverable runs

## Project Structure

```

src/mlplayground/
├── datasets/          # Dataset loaders and abstractions
├── experiments/       # Individual ML experiments
├── models/            # Reusable model factories
├── metrics/           # Evaluation helpers
├── visualization/     # Plotting and analysis utilities
└── core/              # Shared protocols and project paths

data/
├── raw/               # Original datasets
└── bronze/            # Prepared datasets

````

## Included Experiments

### Cobb-Douglas Productivity Dataset

A regression playground based on labor productivity data.

Implemented experiments include:

- Linear regression
- Polynomial regression
- Decision trees
- Random forests
- Support vector regression
- RANSAC regression
- Lasso regularization
- Lasso path analysis
- Hyperparameter search
- Cross-validation strategies

### Iris Dataset

Classification and exploratory analysis examples:

- Feature exploration
- Correlation analysis
- Scatter visualizations
- Support Vector Classification
- Cross-validation approaches

## Installation

The project uses `uv` for dependency management.

Clone the repository:

```bash
git clone https://github.com/avtomatik/ml-playground
cd ml-playground
````

Install dependencies:

```bash
uv sync
```

## Running

List available registered experiments:

```bash
uv run python -m mlplayground
```

Run an experiment directly:

```bash
uv run python -m mlplayground.experiments.cobb_douglas.linear
```

## Testing

Run the test suite:

```bash
uv run pytest
```

The tests verify:

* dataset loading contracts
* experiment execution
* experiment registration
* core protocol definitions

## Design Principles

The project follows a few simple ideas:

* **Datasets provide a consistent interface**
  Each dataset exposes loading, feature extraction, and target extraction.

* **Experiments are isolated and reproducible**
  Each experiment focuses on one modelling approach.

* **Models and metrics are reusable**
  Common estimators and evaluation utilities are separated from experiments.

* **Visualization supports understanding**
  Plots are treated as part of model exploration rather than as an afterthought.

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* pyarrow
* uv

## License

See [LICENSE.md](LICENSE.md).
