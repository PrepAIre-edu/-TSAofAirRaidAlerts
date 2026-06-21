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
df = pd.read_csv("data/official_data_en.csv")

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
    iso = frame["started_at"].dt.isocalendar()
    frame["year_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


# 1.7 підсумкова статистика
print("\n=== ПІДСУМОК ПІСЛЯ ОЧИЩЕННЯ ===")
print(f"Період даних: {df_clean['started_at'].min().date()} "
      f"— {df_clean['started_at'].max().date()}")
print(f"Кількість унікальних областей: {df_clean['oblast'].nunique()}")
print(f"Всього тривог (oblast рівень): {len(df_clean)}")
print(f"\nСередня тривалість: {df_duration['duration_min'].mean():.1f} хв")
print(f"Медіанна тривалість: {df_duration['duration_min'].median():.1f} хв")
print(f"Максимальна тривалість: {df_duration['duration_min'].max():.1f} хв")


# ============================================================
# СЕКЦІЯ 2: ДОСЛІДНИЦЬКИЙ АНАЛІЗ (EDA)
# ============================================================

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# --- Налаштування стилю ---
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#2e3140",
    "axes.labelcolor":  "#c9cdd8",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#7a7f91",
    "ytick.color":      "#7a7f91",
    "text.color":       "#c9cdd8",
    "grid.color":       "#1e2130",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "figure.dpi":       150,
})

ACCENT   = "#e05c5c"   # тривожний червоний
ACCENT2  = "#5c8fe0"   # спокійний синій
MUTED    = "#7a7f91"
BG       = "#0f1117"
PANEL    = "#161b27"

os.makedirs("outputs/plots", exist_ok=True)

def save_and_show(fig, filename):
    path = f"outputs/plots/{filename}"
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"  збережено → {path}")


# ============================================================
# ГРАФІК 1: Тижнева кількість тривог — тренд у часі
# Питання: Як змінювалась інтенсивність атак з часом?
# ============================================================
print("\n[1/5] Тижнева інтенсивність тривог...")

weekly = (
    df_clean
    .groupby("year_week")
    .size()
    .reset_index(name="count")
)

# Щоб правильно сортувати — конвертуємо назад у дату
weekly["date"] = pd.to_datetime(
    weekly["year_week"] + "-1", format="%G-W%V-%u"
)
weekly = weekly.sort_values("date").reset_index(drop=True)

# Ковзне середнє для тренду (4 тижні)
weekly["rolling_4w"] = weekly["count"].rolling(4, center=True).mean()

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)

ax.fill_between(weekly["date"], weekly["count"],
                alpha=0.25, color=ACCENT, linewidth=0)
ax.plot(weekly["date"], weekly["count"],
        color=ACCENT, linewidth=0.8, alpha=0.7, label="Тижнева кількість")
ax.plot(weekly["date"], weekly["rolling_4w"],
        color="#ffffff", linewidth=2, label="Ковзне середнє (4 тижні)")

ax.set_title("Тижнева кількість тривог (oblast рівень)", fontsize=14,
             fontweight="bold", pad=14)
ax.set_xlabel("Дата")
ax.set_ylabel("Кількість тривог")
ax.legend(framealpha=0.1, labelcolor="white")
ax.yaxis.grid(True)
ax.set_axisbelow(True)

fig.tight_layout()
save_and_show(fig, "01_weekly_intensity.png")


# ============================================================
# ГРАФІК 2: Теплова карта — година доби × день тижня
# Питання: Коли протягом тижня тривоги найбільш імовірні?
# ============================================================
print("\n[2/5] Теплова карта година × день тижня...")

DAY_ORDER = ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
UA_DAYS   = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]

heatmap_data = (
    df_clean
    .groupby(["day_of_week", "hour"])
    .size()
    .unstack(fill_value=0)
    .reindex(DAY_ORDER)
)
heatmap_data.index = UA_DAYS

fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="#0f1117",
    annot=False,
    cbar_kws={"label": "Кількість тривог", "shrink": 0.8},
)

ax.set_title("Розподіл тривог: година доби × день тижня",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Година доби (київський час)")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)

# Підпис кожної 4ї години для читабельності
hour_labels = [str(h) if h % 4 == 0 else "" for h in range(24)]
ax.set_xticklabels(hour_labels)

fig.tight_layout()
save_and_show(fig, "02_heatmap_hour_day.png")


# ============================================================
# ГРАФІК 3: Кумулятивний час під тривогою по областях
# Питання: Які регіони несуть найбільше хронічне навантаження?
# ============================================================
print("\n[3/5] Кумулятивний час по областях...")

cumulative = (
    df_duration
    .groupby("oblast")["duration_min"]
    .sum()
    .div(60)                    # конвертуємо в години
    .sort_values(ascending=True)
    .reset_index()
)
cumulative.columns = ["oblast", "hours"]

fig, ax = plt.subplots(figsize=(10, max(6, len(cumulative) * 0.35)))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

# Градієнт кольору по значенню
colors = [ACCENT if h >= cumulative["hours"].quantile(0.75)
          else ACCENT2 for h in cumulative["hours"]]

bars = ax.barh(cumulative["oblast"], cumulative["hours"],
               color=colors, height=0.7)

# Підписи значень
for bar, val in zip(bars, cumulative["hours"]):
    ax.text(val + cumulative["hours"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f} год", va="center", fontsize=7.5, color="#c9cdd8")

ax.set_title("Кумулятивний час під тривогою по областях",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Сумарний час під тривогою (годин)")
ax.xaxis.grid(True)
ax.set_axisbelow(True)

# Легенда
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=ACCENT,  label="Топ 25% навантаження"),
    Patch(facecolor=ACCENT2, label="Решта областей"),
]
ax.legend(handles=legend_elements, framealpha=0.1, labelcolor="white",
          loc="lower right")

fig.tight_layout()
save_and_show(fig, "03_cumulative_hours_by_oblast.png")


# ============================================================
# ГРАФІК 4: Box plot — тривалість тривог по областях
# Питання: Чи однакова типова тривалість в різних регіонах?
# ============================================================
print("\n[4/5] Box plot тривалості по областях...")

# Сортуємо за медіаною для читабельності
median_order = (
    df_duration
    .groupby("oblast")["duration_min"]
    .median()
    .sort_values(ascending=True)
    .index.tolist()
)

fig, ax = plt.subplots(figsize=(10, max(6, len(median_order) * 0.35)))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

# Будуємо boxplot вручну через matplotlib для кращого контролю стилю
data_by_oblast = [
    df_duration[df_duration["oblast"] == ob]["duration_min"].values
    for ob in median_order
]

bp = ax.boxplot(
    data_by_oblast,
    vert=False,
    patch_artist=True,
    labels=median_order,
    flierprops=dict(marker="o", markersize=2,
                    markerfacecolor=ACCENT, alpha=0.3, linestyle="none"),
    medianprops=dict(color="#ffffff", linewidth=1.5),
    boxprops=dict(facecolor=ACCENT2, alpha=0.4, linewidth=0.8),
    whiskerprops=dict(color=MUTED, linewidth=0.8),
    capprops=dict(color=MUTED, linewidth=0.8),
)

ax.set_title("Розподіл тривалості тривог по областях",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Тривалість тривоги (хвилин)")
ax.set_xlim(left=0)
ax.xaxis.grid(True)
ax.set_axisbelow(True)

# Вертикальна лінія — медіана по всій країні
overall_median = df_duration["duration_min"].median()
ax.axvline(overall_median, color=ACCENT, linestyle="--",
           linewidth=1, alpha=0.7, label=f"Медіана по країні ({overall_median:.0f} хв)")
ax.legend(framealpha=0.1, labelcolor="white")

fig.tight_layout()
save_and_show(fig, "04_duration_boxplot_by_oblast.png")


# ============================================================
# ГРАФІК 5: Середня тривалість тривог по місяцях
# Питання: Чи змінювалась типова тривалість тривог з часом?
# ============================================================
print("\n[5/5] Середня тривалість по місяцях...")

monthly_duration = (
    df_duration
    .groupby(["year", "month"])["duration_min"]
    .agg(mean="mean", median="median", count="count")
    .reset_index()
)
monthly_duration["date"] = pd.to_datetime(
    monthly_duration[["year", "month"]].assign(day=1)
)
monthly_duration = monthly_duration.sort_values("date")

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

ax.fill_between(monthly_duration["date"], monthly_duration["mean"],
                alpha=0.15, color=ACCENT2)
ax.plot(monthly_duration["date"], monthly_duration["mean"],
        color=ACCENT2, linewidth=1.5, marker="o", markersize=4,
        label="Середня тривалість")
ax.plot(monthly_duration["date"], monthly_duration["median"],
        color="#ffffff", linewidth=1.5, linestyle="--", marker="s",
        markersize=3, label="Медіанна тривалість")

ax.set_title("Динаміка тривалості тривог по місяцях",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Місяць")
ax.set_ylabel("Тривалість (хвилин)")
ax.legend(framealpha=0.1, labelcolor="white")
ax.yaxis.grid(True)
ax.set_axisbelow(True)

# Додаємо кількість тривог як другу вісь для контексту
ax2 = ax.twinx()
ax2.bar(monthly_duration["date"], monthly_duration["count"],
        width=20, alpha=0.15, color=MUTED, label="Кількість тривог")
ax2.set_ylabel("Кількість тривог", color=MUTED)
ax2.tick_params(axis="y", colors=MUTED)
ax2.spines["right"].set_color("#2e3140")

fig.tight_layout()
save_and_show(fig, "05_monthly_duration_trend.png")


# ============================================================
# ПІДСУМОК EDA
# ============================================================
print("\n=== ПІДСУМОК СЕКЦІЇ 2 ===")
print(f"Пік тижневої активності: {weekly.loc[weekly['count'].idxmax(), 'date'].date()} "
      f"({weekly['count'].max()} тривог)")
print(f"Найнавантаженіша область: {cumulative.iloc[-1]['oblast']} "
      f"({cumulative.iloc[-1]['hours']:,.0f} год)")
print(f"Найменш навантажена область: {cumulative.iloc[0]['oblast']} "
      f"({cumulative.iloc[0]['hours']:,.0f} год)")
print(f"Медіанна тривалість по країні: {overall_median:.0f} хв")
print("\nГрафіки збережено у outputs/plots/")

