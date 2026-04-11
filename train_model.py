import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # Загрузка данных
    df = pd.read_csv("dataset.csv")

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

    # Сохранение модели
    joblib.dump(model, "price_prediction_model.pkl")
    print("\nМодель сохранена в файл: price_prediction_model.pkl")


if __name__ == "__main__":
    main()