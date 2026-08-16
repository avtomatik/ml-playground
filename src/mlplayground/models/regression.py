from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def linear_regression():
    return LinearRegression()


def random_forest():
    return RandomForestRegressor(n_estimators=1000, random_state=1, n_jobs=-1)


def svr():
    return make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
