from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Загрузим данные
data_set = pd.read_csv(DATA_PATH)

# Диаграмма рассеяния Площадь vs Цена
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='площадь m^2',
    y='цена',
    data=data_set,
    color='green',
    s=50,
    alpha=0.6
)
plt.title('Зависимость площади от цены', fontsize=15)

plt.xlabel('Площадь (м²)', fontsize=12)
plt.ylabel('Цена (руб.)', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(FIGURES_DIR / "area_price_dependency.png", dpi=300, bbox_inches="tight")

plt.show()