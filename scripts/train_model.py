from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "price_prediction_model.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.txt"


def relative_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_pred = np.clip(y_pred, 0, None)

    return np.abs(y_true - y_pred) / y_true


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, None)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "name": name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def find_best_knn_by_loo(X_train, y_train):
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(weights="distance"))
    ])

    parameters = {
        "knn__n_neighbors": [2, 3, 5, 7, 9, 11, 15, 21]
    }

    def loo_error_score(estimator, X, y):
        predictions = estimator.predict(X)
        errors = relative_error(y, predictions)
        return -errors.mean()

    search = GridSearchCV(
        estimator=knn_pipeline,
        param_grid=parameters,
        scoring=loo_error_score,
        cv=LeaveOneOut(),
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    optimal_k = search.best_params_["knn__n_neighbors"]
    minimum_loo_error = -search.best_score_

    return search.best_estimator_, optimal_k, minimum_loo_error


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["цена"])
    y = df["цена"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    best_knn_model, optimal_k, minimum_loo_error = find_best_knn_by_loo(
        X_train,
        y_train
    )

    models = [
        (
            "Linear Regression",
            LinearRegression()
        ),
        (
            "KNN Regressor",
            best_knn_model
        )
    ]

    results = []

    for name, model in models:
        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )
        results.append(result)

    best_result = min(
        results,
        key=lambda item: item["rmse"]
    )

    best_model = best_result["model"]

    metrics_lines = [
        "Model evaluation",
        "",
        "Models:",
        "Linear Regression",
        "KNN Regressor",
        "",
        "KNN validation:",
        f"Optimal k by Leave-One-Out: {optimal_k}",
        f"Minimum LOO relative error: {minimum_loo_error:.4f}",
        "",
        "Regression metrics on test set:",
    ]

    for result in results:
        metrics_lines.extend([
            "",
            f"Model: {result['name']}",
            f"MAE:  {result['mae']:,.2f}",
            f"RMSE: {result['rmse']:,.2f}",
            f"R2:   {result['r2']:.4f}",
        ])

    metrics_lines.extend([
        "",
        f"Best model by RMSE: {best_result['name']}"
    ])

    metrics_text = "\n".join(metrics_lines)

    print(metrics_text)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nЛучшая модель сохранена в файл: {MODEL_PATH}")


if __name__ == "__main__":
    main()