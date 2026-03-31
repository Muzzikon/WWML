import pandas as pd
import numpy as np
import random

def generate_apartment():
    location = random.randint(1, 10)  # Район
    rooms = random.randint(1, 6)  # Количество комнат
    distance = round(random.uniform(0.5, 25), 2)  # Удаленность от центра (км)
    floor = random.randint(1, 25)  # Этаж

    mean_area = 50  # Средняя площадь
    std_area = 20   # Стандартное отклонение площади
    area = round(np.random.normal(mean_area, std_area), 1)
    area = max(area, 10)  # Площадь не может быть меньше 10 м²

    base_m2_price = 150000  # Цена за м²
    loc_coeff = 1.7 - (location / 10)  # Коэффициент района
    dist_coeff = max(0.6, 1 - (distance / 50))  # Коэффициент удаленности
    final_m2_price = base_m2_price * loc_coeff * dist_coeff

    total_price = ((int((area * final_m2_price) * random.uniform(0.95, 1.05)) // 10000) * 10000)

    total_vis_price = str(total_price)[:-6] + '.' + str(total_price)[-6:-3] + '.' + str(total_price)[-4:-1]

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
    data_set.to_csv('dataset.csv', index=False)  # Сохраняем данные в CSV файл