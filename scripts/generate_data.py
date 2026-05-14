from pathlib import Path

import pandas as pd
import numpy as np
import random


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"


def generate_apartment():
    location = random.randint(1, 10)  # Район
    rooms = random.randint(1, 6)  # Количество комнат
    distance = round(random.uniform(0.5, 25), 2)  # Удаленность от центра (км)
    floor = random.randint(1, 25)  # Этаж

    mean_area_by_rooms = {
        1: 38,
        2: 55,
        3: 75,
        4: 95,
        5: 115,
        6: 135
    }

    mean_area = mean_area_by_rooms[rooms]
    std_area = 12

    area = round(np.random.normal(mean_area, std_area), 1)
    area = max(area, 15)

    base_m2_price = 150000  # Цена за м²
    loc_coeff = 1.7 - (location / 10)  # Коэффициент района
    dist_coeff = max(0.6, 1 - (distance / 50))  # Коэффициент удаленности

    if floor == 1:
        floor_coeff = 0.92
    elif 2 <= floor <= 5:
        floor_coeff = 1.00
    elif 6 <= floor <= 15:
        floor_coeff = 1.07
    elif 16 <= floor <= 22:
        floor_coeff = 1.04
    else:
        floor_coeff = 0.97

    rooms_coeff = 1 + (rooms - 2) * 0.04

    final_m2_price = base_m2_price * loc_coeff * dist_coeff * floor_coeff * rooms_coeff

    raw_price = area * final_m2_price * random.uniform(0.95, 1.05)
    total_price = (int(raw_price) // 10000) * 10000

    information = {
        "район": location,
        "площадь m^2": area,
        "кол-во комнат": rooms,
        "удаленность от центра (км)": distance,
        "этаж": floor,
        "цена": total_price
    }
    return information


def generate_data(n=1000):
    data = [generate_apartment() for _ in range(n)]
    return pd.DataFrame(data)


if __name__ == "__main__":
    data_set = generate_data()

    # Сохраняем данные в CSV файл
    DATA_PATH.parent.mkdir(exist_ok=True)
    data_set.to_csv(DATA_PATH, index=False)
    print(f"Датасет сохранён: {DATA_PATH}")