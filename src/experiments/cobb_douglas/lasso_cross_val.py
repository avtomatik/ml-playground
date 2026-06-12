import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

from datasets.cobb_douglas import load as load_cobb_douglas


def compare_r2s_print_out_coefs(X: np.ndarray, y: np.ndarray) -> None:
    # =========================================================================
    # TODO: Split Function to Increase Cohesion
    # =========================================================================
    solver = LassoCV(cv=4, random_state=0).fit(X, y)
    y_pred = solver.predict(X)

    r2_solver = solver.score(X, y)
    r2_metrics = r2_score(y, y_pred)

    print(solver.coef_)
    print(solver.intercept_)
    print(f"R**2 Powered by sklearn.linear_model.LassoCV: {r2_solver:.6}")
    print(f"R**2 Powered by sklearn.metrics.r2_score: {r2_metrics:.6}")


if __name__ == "__main__":
    # =========================================================================
    # Make Dataset
    # =========================================================================
    df = load_cobb_douglas()

    X = df[["labor_capital_intensity"]]
    y = df["labor_productivity"]

    compare_r2s_print_out_coefs(X, y)
