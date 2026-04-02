from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd

# Обучение модели
data_set = pd.read_csv('dataset.csv')

# Разделим данные на признаки и целевую переменную
X = data_set.drop(columns=["цена"])  # Признаки
y = data_set["цена"]  # Целевая переменная

# Разделим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания на тестовых данных
y_pred = model.predict(X_test)

# Оценка качества модели
mae = mean_absolute_error(y_test, y_pred)  # Средняя абсолютная ошибка
r2 = r2_score(y_test, y_pred)  # Коэффициент детерминации

print(f"MAE: {mae}")
print(f"R2: {r2}")

# Сохраняем модель
joblib.dump(model, "price_prediction_model.pkl")