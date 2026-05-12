from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "price_prediction_model.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.txt"


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = predictions.clip(min=0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return {
        "name": name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


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

    models = [
        (
            "Linear Regression",
            LinearRegression()
        ),
        (
            "Decision Tree Regressor",
            DecisionTreeRegressor(
                max_depth=6,
                min_samples_leaf=10,
                random_state=42
            )
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

    best_result = min(results, key=lambda item: item["rmse"])
    best_model = best_result["model"]

    metrics_lines = []

    for result in results:
        metrics_lines.append(f"Model: {result['name']}")
        metrics_lines.append(f"MAE:  {result['mae']:,.2f}")
        metrics_lines.append(f"RMSE: {result['rmse']:,.2f}")
        metrics_lines.append(f"R2:   {result['r2']:.4f}")
        metrics_lines.append("")

    metrics_lines.append(f"Best model: {best_result['name']}")

    metrics_text = "\n".join(metrics_lines)

    print(metrics_text)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nЛучшая модель сохранена в файл: {MODEL_PATH}")


if __name__ == "__main__":
    main()