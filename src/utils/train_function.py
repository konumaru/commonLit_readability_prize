import os
import pickle
from typing import Callable, NoReturn

import numpy as np
import xgboost as xgb
from sklearn import model_selection
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def train_xbg(X, y):
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=10_000,
        max_depth=2,
        learning_rate=0.2,
        objective="reg:squarederror",
        min_child_weight=10,
        subsample=0.8,
        colsample_bylevel=0.2,
        colsample_bynode=0.2,
        reg_alpha=3,
        reg_lambda=3,
        random_state=42,
        # gpu_id=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="rmse",
        early_stopping_rounds=20,
        verbose=10,
    )
    return model


def train_svr(X, y):
    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.1))
    model.fit(X, y)
    return model


def train_ridge(X, y):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    return model


def train_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    train_fn: Callable,
    save_dir: str = "../data/models/tmp/",
    y_cv: np.ndarray = None,
    verbose: bool = True,
) -> NoReturn:
    """Cross validation function.

    Parameters
    ----------
    X : np.ndarray
        Feature data.
    y : np.ndarray
        Target or label data.
    train_fn : Callable
        train function.
    save_dir : str, optional
        Save directory of trained model, by default "../data/models/tmp/"
    n_splits : int, optional
        split number, by default 5
    verbose : bool, optional
        Is print RMSE score, by default True

    Returns
    -------
    NoReturn
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    metrics = []
    for n_fold, (train_idx, test_idx) in enumerate(cv.split(X, y_cv)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = train_fn(X_train, y_train)
        with open(os.path.join(save_dir, f"{n_fold}-fold.pkl"), mode="wb") as file:
            pickle.dump(model, file)

        y_pred = model.predict(X_test)
        metric = mean_squared_error(y_test, y_pred, squared=False)

        metrics.append(metric)

    if verbose:
        print(f"Average RMSE: {np.mean(metrics):.06f} ± {np.std(metrics):.4f}")


def main():
    iris = load_iris()

    data = iris.data
    target = iris.target
    cv = model_selection.KFold(n_splits=5)

    print("train Xgboost:")
    train_cross_validate(data, target, cv, train_xbg, save_dir="../data/models/xgb/")

    print("\ntrain SVR")
    train_cross_validate(data, target, cv, train_svr, save_dir="../data/models/svr/")

    print("\nTrain Ridge")
    train_cross_validate(
        data, target, cv, train_ridge, save_dir="../data/models/ridge/"
    )


if __name__ == "__main__":
    main()
