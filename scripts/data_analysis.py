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

# Генерация гистограммы цен
sns.set_theme(style="ticks", palette="pastel")
plt.figure(figsize=(10, 6))
prices_in_millions = data_set["цена"] / 1_000_000
sns.histplot(prices_in_millions, bins=30, kde=True, color='blue')
plt.title("Распределение квартир по цене", fontsize=15)

plt.xlabel("Цена (млн. руб.)", fontsize=12)
plt.ylabel("Количество квартир", fontsize=12)

plt.ticklabel_format(style='plain', axis='x')
plt.savefig(FIGURES_DIR / "price_distribution.png", dpi=300, bbox_inches="tight")

plt.show()

# Матрица корреляции
correlation_matrix = data_set.corr()
plt.figure(figsize=(10, 8))

sns.heatmap(correlation_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
            fmt='.2f',
            annot_kws={'size': 10})

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")

plt.show()