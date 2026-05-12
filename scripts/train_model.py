from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "price_prediction_model.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.txt"


def main():
    # Загрузка данных
    df = pd.read_csv(DATA_PATH)

    # Признаки и целевая переменная
    X = df.drop(columns=["цена"])
    y = df["цена"]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Улучшенная модель
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    # Обучение
    model.fit(X_train, y_train)

    # Предсказания
    preds = model.predict(X_test)

    # Метрики
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("Improved model: Polynomial(2) + Ridge")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2:   {r2:.4f}")

    metrics_text = (
        "Improved model: Polynomial(2) + Ridge\n"
        f"MAE:  {mae:,.2f}\n"
        f"RMSE: {rmse:,.2f}\n"
        f"R2:   {r2:.4f}\n"
    )

    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    # Сохранение модели
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nМодель сохранена в файл: {MODEL_PATH}")


if __name__ == "__main__":
    main()