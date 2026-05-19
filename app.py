from pathlib import Path

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Оценка стоимости квартиры",
    page_icon="🏠",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "price_prediction_model.pkl"

# Загружаем модель один раз при старте
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.markdown("""
<style>
/* Общий контейнер */
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Карточки */
.custom-card {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 18px;
}

/* Карточка результата */
.result-card {
    background: #172554;
    border: 1px solid #1d4ed8;
    padding: 24px;
    border-radius: 18px;
    margin-top: 20px;
}

.main-card {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 18px;
}

.result-title {
    color: #f9fafb;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
}

.price-text {
    color: #ffffff;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}

.range-text {
    color: #cbd5e1;
    font-size: 15px;
}
            
/* Текст */
.card-title {
    color: #f9fafb;
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-subtitle {
    color: #9ca3af;
    font-size: 15px;
}

.section-title {
    color: #f3f4f6;
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 8px;
}

.big-price {
    color: #ffffff;
    font-size: 30px;
    font-weight: 700;
    margin-top: 10px;
}

.small-text {
    color: #cbd5e1;
    font-size: 15px;
}

/* Кнопка */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    height: 48px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-card">
    <div class="card-title">🏠 Оценка стоимости квартиры</div>
    <div class="card-subtitle">
        Введите параметры квартиры слева и получите ориентировочную рыночную стоимость.
    </div>
</div>
""", unsafe_allow_html=True)

# Поля ввода
with st.sidebar:
    st.header("Параметры квартиры")

    with st.form("apartment_form"):
        location = st.slider(
            "Класс района",
            min_value=1,
            max_value=10,
            value=5,
            help="1 — престижный район, 10 — менее престижный"
        )

        rooms = st.slider(
            "Количество комнат",
            min_value=1,
            max_value=6,
            value=2
        )

        floor = st.slider(
            "Этаж",
            min_value=1,
            max_value=25,
            value=5
        )

        area = st.number_input(
            "Площадь (м²)",
            min_value=15.0,
            max_value=200.0,
            value=50.0,
            step=0.5
        )

        distance = st.number_input(
            "Удалённость от центра (км)",
            min_value=0.5,
            max_value=25.0,
            value=5.0,
            step=0.1
        )

        submitted = st.form_submit_button("Рассчитать стоимость")


st.markdown("""
<div class="main-card">
    <div class="result-title">📋 Выбранные параметры</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.metric("Площадь", f"{area} м²")
    st.metric("Комнаты", rooms)
    st.metric("Этаж", floor)

with col2:
    st.metric("Класс района", location)
    st.metric("Удалённость", f"{distance} км")


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
        st.error(
            "Для выбранного количества комнат площадь слишком маленькая. "
            "Увеличьте площадь квартиры."
        )
        st.stop()

    features = pd.DataFrame([{
        "район": location,
        "площадь m^2": area,
        "кол-во комнат": rooms,
        "удаленность от центра (км)": distance,
        "этаж": floor
    }])

    price = max(model.predict(features)[0], 0)

    min_price = price * 0.997
    max_price = price * 1.003
    avg_price = (min_price + max_price) / 2

    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">💰 Ориентировочная стоимость</div>
        <div class="price-text">{avg_price / 1_000_000:.2f} млн руб.</div>
        <div class="range-text">
            Диапазон оценки: от {int(min_price):,} до {int(max_price):,} руб.
        </div>
    </div>
    """.replace(",", " "), unsafe_allow_html=True)

    st.caption(
        "Прогноз является ориентировочным, так как модель обучена на синтетических данных."
    )
else:
    st.info("Заполните параметры слева и нажмите «Рассчитать стоимость».")
