import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузим данные
data_set = pd.read_csv('dataset.csv')

# Диаграмма рассеяния Площадь vs Цена
plt.figure(figsize=(10,6))
sns.scatterplot(x='площадь m^2', y='цена', data=data_set, color='green', s=50, alpha=0.6)
plt.title('Зависимость площади от цены', fontsize=15)
plt.xlabel('Площадь (м²)', fontsize=12)
plt.ylabel('Цена (руб.)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()