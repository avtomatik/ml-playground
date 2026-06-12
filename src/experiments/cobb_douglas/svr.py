from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from datasets.cobb_douglas import load as load_cobb_douglas

if __name__ == "__main__":
    # =========================================================================
    # Make Dataset
    # =========================================================================
    df = load_cobb_douglas()

    X = df[["labor_capital_intensity"]]
    y = df["labor_productivity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.45, random_state=1
    )

    solver = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    solver.fit(X_train, y_train)

    y_train_pred = solver.predict(X_train)
    y_test_pred = solver.predict(X_test)

    plt.scatter(
        y_train_pred,
        y_train_pred - y_train,
        c="black",
        marker="o",
        s=35,
        alpha=0.5,
        label="Train",
    )
    plt.scatter(
        y_test_pred,
        y_test_pred - y_test,
        c="lightgreen",
        marker="s",
        s=35,
        alpha=0.7,
        label="Test",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.legend(loc="upper right")
    plt.hlines(y=0, xmin=y.min(), xmax=y.max(), lw=2, color="red")
    plt.xlim([y.min(), y.max()])
    plt.grid()
    plt.show()
    print(
        f"MSE on Train Data: {mean_squared_error(y_train, y_train_pred):,.4f}"
    )
    print(f"MSE on Test Data: {mean_squared_error(y_test, y_test_pred):,.4f}")
    print(f"R**2 on Train Data: {r2_score(y_train, y_train_pred):,.4f}")
    print(f"R**2 on Test Data: {r2_score(y_test, y_test_pred):,.4f}")
