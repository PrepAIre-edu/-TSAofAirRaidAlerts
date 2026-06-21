# ============================================================
# ПОВІТРЯНІ ТРИВОГИ В УКРАЇНІ — АНАЛІЗ ЧАСОВИХ РЯДІВ
# ============================================================
# Застереження: всі патерни та прогнози базуються виключно
# на історичних даних про оповіщення. Вони не відображають
# реальні загрози і не є оперативними передбаченнями.
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# СЕКЦІЯ 1: ЗАВАНТАЖЕННЯ ТА ОЧИЩЕННЯ ДАНИХ
# ============================================================

# --- 1.1 Завантаження ---
df = pd.read_csv("data/alerts.csv")

print("=== БАЗОВА ІНФОРМАЦІЯ ===")
print(f"Всього записів: {len(df)}")
print(f"\nКолонки: {df.columns.tolist()}")
print(f"\nПерші 3 рядки:")
print(df.head(3))


# --- 1.2 Конвертація дат і часових поясів ---
df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

# Конвертуємо в київський час — важливо для аналізу годин доби
df["started_at"] = df["started_at"].dt.tz_convert("Europe/Kyiv")
df["finished_at"] = df["finished_at"].dt.tz_convert("Europe/Kyiv")


# --- 1.3 Фільтрація по рівню та джерелу ---
print("\n=== РОЗПОДІЛ ПО РІВНЯХ (level) ===")
print(df["level"].value_counts())

print("\n=== РОЗПОДІЛ ПО ДЖЕРЕЛАХ (source) ===")
print(df["source"].value_counts())

# Залишаємо лише oblast-рівень і офіційні записи
# щоб уникнути дублікатів і ненадійних даних
df_clean = df[
    (df["level"] == "oblast") &
    (df["source"] == "official")
].copy()

print(f"\nПісля фільтрації: {len(df_clean)} записів")
print(f"Відфільтровано: {len(df) - len(df_clean)} записів")


# --- 1.4 Пропущені значення ---
print("\n=== ПРОПУЩЕНІ ЗНАЧЕННЯ ===")
missing = df_clean.isnull().sum()
print(missing[missing > 0] if missing.any() else "Пропущених значень немає")

missing_finish = df_clean["finished_at"].isnull().sum()
print(f"\nТривог без finished_at: {missing_finish} "
      f"({missing_finish/len(df_clean)*100:.1f}%)")


# --- 1.5 Розрахунок тривалості ---
# Рахуємо тільки там де є обидві дати
df_clean["duration_min"] = (
    (df_clean["finished_at"] - df_clean["started_at"])
    .dt.total_seconds() / 60
)

# Перевірка аномальних тривалостей
print("\n=== АНОМАЛЬНА ТРИВАЛІСТЬ ===")
too_short = df_clean["duration_min"] < 1
too_long = df_clean["duration_min"] > 1440  # більше 24 годин
negative = df_clean["duration_min"] < 0

print(f"Менше 1 хвилини: {too_short.sum()}")
print(f"Більше 24 годин: {too_long.sum()}")
print(f"Від'ємна тривалість: {negative.sum()}")

# Позначаємо підозрілі записи окремим флагом — не видаляємо
df_clean["duration_suspicious"] = too_short | too_long | negative

# Окремий датафрейм для аналізу тривалості — без підозрілих і без NaN
df_duration = df_clean[
    (~df_clean["duration_suspicious"]) &
    (df_clean["duration_min"].notna())
].copy()

print(f"\nЗаписів придатних для аналізу тривалості: {len(df_duration)}")


# 1.6 часові ознаки
for frame in [df_clean, df_duration]:
    frame["date"] = frame["started_at"].dt.date
    frame["hour"] = frame["started_at"].dt.hour
    frame["day_of_week"] = frame["started_at"].dt.day_name()
    frame["month"] = frame["started_at"].dt.month
    frame["year"] = frame["started_at"].dt.year
    frame["week"] = frame["started_at"].dt.isocalendar().week.astype(int)
    frame["year_week"] = (
        frame["started_at"].dt.strftime("%Y-W%W")
    )


# 1.7 підсумкова статистика
print("\n=== ПІДСУМОК ПІСЛЯ ОЧИЩЕННЯ ===")
print(f"Період даних: {df_clean['started_at'].min().date()} "
      f"— {df_clean['started_at'].max().date()}")
print(f"Кількість унікальних областей: {df_clean['oblast'].nunique()}")
print(f"Всього тривог (oblast рівень): {len(df_clean)}")
print(f"\nСередня тривалість: {df_duration['duration_min'].mean():.1f} хв")
print(f"Медіанна тривалість: {df_duration['duration_min'].median():.1f} хв")
print(f"Максимальна тривалість: {df_duration['duration_min'].max():.1f} хв")
