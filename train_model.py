from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd

# Обучение модели
data_set = pd.read_csv('dataset.csv')

# Функция обучения и оценки модели
def train_and_evaluate(data, model_name):
    X = data.drop(columns=["цена"])
    y = data["цена"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name}")
    print(f"Количество строк: {len(data)}")
    print(f"MAE: {mae:,.2f}")
    print(f"R2: {r2:.4f}")

    return model, mae, r2

# 1. Базовая модель
base_model, base_mae, base_r2 = train_and_evaluate(data_set, "Базовая модель")

# 2. Очистка выбросов по IQR для цены и площади
cleaned_data = data_set.copy()

for column in ["цена", "площадь m^2"]:
    Q1 = cleaned_data[column].quantile(0.25)
    Q3 = cleaned_data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cleaned_data = cleaned_data[
        (cleaned_data[column] >= lower_bound) &
        (cleaned_data[column] <= upper_bound)
    ]

# 3. Модель после очистки выбросов
clean_model, clean_mae, clean_r2 = train_and_evaluate(
    cleaned_data, "Модель после удаления выбросов"
)

# 4. Сравнение и сохранение лучшей модели
if clean_r2 > base_r2:
    best_model = clean_model
    print("\nЛучшая модель: после удаления выбросов")
else:
    best_model = base_model
    print("\nЛучшая модель: базовая")

joblib.dump(best_model, "price_prediction_model.pkl")
print("Лучшая модель сохранена в файл price_prediction_model.pkl")