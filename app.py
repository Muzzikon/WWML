from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "price_prediction_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def format_price(price):
    rounded_price = round(price / 10_000) * 10_000
    return f"{int(rounded_price):,}".replace(",", " ")


st.set_page_config(
    page_title="Оценка стоимости квартиры",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

model = load_model()

st.title("🏠 Оценка стоимости квартиры")
st.write("Введите параметры квартиры для ориентировочной оценки.")

with st.form("apartment_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Площадь, м²",
            min_value=15.0,
            max_value=200.0,
            value=50.0,
            step=0.5
        )

        rooms = st.slider(
            "Комнат",
            min_value=1,
            max_value=6,
            value=2
        )

        location = st.slider(
            "Класс района",
            min_value=1,
            max_value=10,
            value=5
        )

    with col2:
        distance = st.number_input(
            "Удалённость от центра, км",
            min_value=0.5,
            max_value=25.0,
            value=5.0,
            step=0.1
        )

        floor = st.slider(
            "Этаж",
            min_value=1,
            max_value=25,
            value=5
        )

        submitted = st.form_submit_button("Оценить")


if submitted:
    min_area_by_rooms = {
        1: 20,
        2: 35,
        3: 50,
        4: 70,
        5: 90,
        6: 110
    }

    if area < min_area_by_rooms[rooms]:
        st.error("Для выбранного количества комнат площадь слишком маленькая.")
        st.stop()

    features = pd.DataFrame([{
        "район": location,
        "площадь m^2": area,
        "кол-во комнат": rooms,
        "удаленность от центра (км)": distance,
        "этаж": floor
    }])

    price = model.predict(features)[0]
    price = max(price, 0)

    formatted_price = format_price(price)

    st.markdown("**Результат оценки**")
    st.metric(
        label="Ориентировочная стоимость квартиры",
        value=f"{formatted_price} руб."
    )