import streamlit as st
import joblib
import pandas as pd

# Загружаем обученную модель
model = joblib.load(r"D:\WWML\price_prediction_model.pkl")

st.title("Оценка стоимости квартиры")

# Ввод данных пользователем
location = st.number_input("Район (1-10)", min_value=1, max_value=10, value=5)
area = st.number_input("Площадь (м^2)", min_value=10, max_value=300, value=50)
rooms = st.number_input("Количество комнат", min_value=1, max_value=10, value=2)
distance = st.number_input("Удаленность от центра (км)", min_value=1, max_value=50, value=5)
floor = st.number_input("Этаж", min_value=1, max_value=25, value=1)

if st.button("Оценить"):
    input_data = pd.DataFrame([{
        "район": location,
        "площадь m^2": area,
        "кол-во комнат": rooms,
        "удаленность от центра (км)": distance,
        "этаж": floor
    }])

    predicted_price = model.predict(input_data)[0]

    st.write("Введенные данные:")
    st.dataframe(input_data)

    st.success(f"Предсказанная стоимость квартиры: {predicted_price:,.2f} рублей")