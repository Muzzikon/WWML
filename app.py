import streamlit as st
import joblib
import numpy as np

# Загружаем модель один раз при старте
@st.cache_resource
def load_model():
    return joblib.load("price_prediction_model.pkl")

model = load_model()

st.title("🏠 Оценка стоимости квартиры")
st.write("Введите параметры квартиры — получите предсказанную цену.")

# Поля ввода
col1, col2 = st.columns(2)

with col1:
    location = st.slider("Район (1 = центр, 10 = окраина)", 1, 10, 5)
    rooms = st.slider("Количество комнат", 1, 6, 2)
    floor = st.slider("Этаж", 1, 25, 5)

with col2:
    area = st.number_input("Площадь (м²)", min_value=15.0, max_value=200.0, value=50.0, step=0.5)
    distance = st.number_input("Удалённость от центра (км)", min_value=0.5, max_value=25.0, value=5.0, step=0.1)

# Предсказание
if st.button("Рассчитать цену"):
    features = np.array([[location, area, rooms, distance, floor]])
    price = model.predict(features)[0]
    
    # Красивый вывод
    price_millions = price / 1_000_000
    st.success(f"💰 Предсказанная цена: **{price_millions:.2f} млн. руб.**")
    st.caption(f"({int(price):,} руб.)".replace(",", " "))