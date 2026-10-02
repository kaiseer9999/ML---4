import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================================

df = pd.read_csv("data/data.csv")

print("=" * 60)
print("ПРАКТИЧЕСКАЯ РАБОТА №4")
print("=" * 60)

print("\nРазмер данных:")
print(df.shape)

print("\nСтолбцы:")
print(df.columns.tolist())


# ==========================================
# 2. РАЗДЕЛЕНИЕ НА X И y
# ==========================================

target = "Median_House_Value"

X = df.drop(columns=[target])
y = df[target]

print("\nЦелевая переменная:")
print(target)

print("\nКоличество признаков:", X.shape[1])


# ==========================================
# 3. РАЗДЕЛЕНИЕ НА TRAIN / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nРазмер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)


# ==========================================
# 4. БАЗОВАЯ МОДЕЛЬ
# ==========================================

model_base = LinearRegression()

model_base.fit(X_train, y_train)

y_pred_base = model_base.predict(X_test)


# ==========================================
# 5. МЕТРИКИ БАЗОВОЙ МОДЕЛИ
# ==========================================

mae_base = mean_absolute_error(y_test, y_pred_base)

mse_base = mean_squared_error(y_test, y_pred_base)

rmse_base = np.sqrt(mse_base)

r2_base = r2_score(y_test, y_pred_base)

print("\n" + "=" * 60)
print("БАЗОВАЯ МОДЕЛЬ")
print("=" * 60)

print(f"MAE  : {mae_base:.2f}")
print(f"MSE  : {mse_base:.2f}")
print(f"RMSE : {rmse_base:.2f}")
print(f"R²   : {r2_base:.4f}")


# ==========================================
# 6. КОРРЕЛЯЦИОННАЯ МАТРИЦА
# ==========================================

plt.figure(figsize=(14, 10))

correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Корреляционная матрица признаков")

plt.tight_layout()

plt.savefig("correlation_matrix.png", dpi=300)

plt.show()


# ==========================================
# 7. VIF — ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ
# ==========================================

print("\n" + "=" * 60)
print("VIF — МУЛЬТИКОЛЛИНЕАРНОСТЬ")
print("=" * 60)

vif_data = pd.DataFrame()

vif_data["Feature"] = X.columns

vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif_data = vif_data.sort_values(
    by="VIF",
    ascending=False
)

print(vif_data.to_string(index=False))


# ==========================================
# 8. УДАЛЕНИЕ ИЗБЫТОЧНОГО ПРИЗНАКА
# ==========================================

# На основании расстояний до городов
# некоторые признаки могут сильно коррелировать.
#
# Для сокращенной модели убираем
# Distance_to_SanDiego как потенциально избыточный признак.

removed_feature = "Distance_to_SanDiego"

X_reduced = X.drop(columns=[removed_feature])

print("\n" + "=" * 60)
print("СОКРАЩЕННАЯ МОДЕЛЬ")
print("=" * 60)

print("Удаленный признак:", removed_feature)

print("\nОставшиеся признаки:")

for feature in X_reduced.columns:
    print("-", feature)


# ==========================================
# 9. TRAIN / TEST ДЛЯ СОКРАЩЕННОЙ МОДЕЛИ
# ==========================================

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 10. ОБУЧЕНИЕ СОКРАЩЕННОЙ МОДЕЛИ
# ==========================================

model_reduced = LinearRegression()

model_reduced.fit(
    X_train_red,
    y_train_red
)

y_pred_reduced = model_reduced.predict(
    X_test_red
)


# ==========================================
# 11. МЕТРИКИ СОКРАЩЕННОЙ МОДЕЛИ
# ==========================================

mae_reduced = mean_absolute_error(
    y_test_red,
    y_pred_reduced
)

mse_reduced = mean_squared_error(
    y_test_red,
    y_pred_reduced
)

rmse_reduced = np.sqrt(mse_reduced)

r2_reduced = r2_score(
    y_test_red,
    y_pred_reduced
)

print("\nМетрики сокращенной модели:")

print(f"MAE  : {mae_reduced:.2f}")
print(f"MSE  : {mse_reduced:.2f}")
print(f"RMSE : {rmse_reduced:.2f}")
print(f"R²   : {r2_reduced:.4f}")


# ==========================================
# 12. СРАВНЕНИЕ МОДЕЛЕЙ
# ==========================================

comparison = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R²"
    ],
    "Base model": [
        mae_base,
        mse_base,
        rmse_base,
        r2_base
    ],
    "Reduced model": [
        mae_reduced,
        mse_reduced,
        rmse_reduced,
        r2_reduced
    ]
})

print("\n" + "=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

print(comparison.to_string(index=False))


# ==========================================
# 13. ГРАФИК ОСТАТКОВ
# ==========================================

residuals = y_test_red - y_pred_reduced

plt.figure(figsize=(10, 6))

plt.scatter(
    y_pred_reduced,
    residuals,
    alpha=0.4
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Предсказанные значения")

plt.ylabel("Остатки")

plt.title("График остатков сокращенной модели")

plt.tight_layout()

plt.savefig(
    "residuals.png",
    dpi=300
)

plt.show()


# ==========================================
# 14. ИТОГ
# ==========================================

print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)

print(
    "\nБазовая модель использует все 13 доступных признаков."
)

print(
    f"R² базовой модели: {r2_base:.4f}"
)

print(
    f"R² сокращенной модели: {r2_reduced:.4f}"
)

print(
    f"\nИз модели был удален признак: {removed_feature}"
)

if r2_reduced >= r2_base:
    print(
        "Сокращенная модель сохранила или улучшила качество "
        "по R² при меньшем количестве признаков."
    )
else:
    print(
        "После удаления признака качество модели немного снизилось."
    )

print(
    "\nГрафик корреляции сохранен в: correlation_matrix.png"
)

print(
    "График остатков сохранен в: residuals.png"
)