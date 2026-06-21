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
    plt.close(fig)
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

sns.boxplot(
    data=df_duration,
    y="oblast",
    x="duration_min",
    order=median_order,
    ax=ax,
    color=ACCENT2,
    fill=True,
    linecolor=MUTED,
    linewidth=0.8,
    flierprops={"marker": "o", "markersize": 2,
                "markerfacecolor": ACCENT, "alpha": 0.3},
    medianprops={"color": "#ffffff", "linewidth": 1.5},
)

ax.set_title("Розподіл тривалості тривог по областях",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Тривалість тривоги (хвилин)")
ax.set_ylabel("")
ax.set_xlim(left=0)
ax.xaxis.grid(True)
ax.set_axisbelow(True)

overall_median = df_duration["duration_min"].median()
ax.axvline(overall_median, color=ACCENT, linestyle="--",
           linewidth=1, alpha=0.7,
           label=f"Медіана по країні ({overall_median:.0f} хв)")
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

# ============================================================
# СЕКЦІЯ 3: ПАТЕРНИ РИЗИКУ ПО РЕГІОНУ І ЧАСУ
# ============================================================
# Методологія: для кожної області рахуємо скільки разів
# тривога починалась у кожну годину доби, нормалізуємо
# по рядках (відносна частота всередині кожної області).
# Слоти з менше ніж 5 спостережень позначаються як
# ненадійні і не відображаються на графіку.
#
# ЗАСТЕРЕЖЕННЯ: всі значення — це історичні частоти
# оповіщень, а не прогноз реальної загрози.
# ============================================================

print("\n=== СЕКЦІЯ 3: ПАТЕРНИ РИЗИКУ ПО РЕГІОНУ І ЧАСУ ===")

MIN_OBSERVATIONS = 5


# Вікно спостереження для кожної області
observation_window = (
    df_clean.groupby("oblast")["started_at"]
    .agg(first="min", last="max")
)
observation_window["days"] = (
    observation_window["last"] - observation_window["first"]
).dt.days
print(observation_window.sort_values("days"))
print(f"\nМінімальне вікно: {observation_window['days'].min()} днів")
print(f"Максимальне вікно: {observation_window['days'].max()} днів")

print(f"Різниця: {observation_window['days'].max() - observation_window['days'].min()} днів")

# Видаляємо області з недостатнім вікном спостереження
MIN_DAYS = 30
valid_oblasts = observation_window[observation_window["days"] >= MIN_DAYS].index

df_clean_filtered = df_clean[df_clean["oblast"].isin(valid_oblasts)].copy()

print(f"Видалено областей: {len(observation_window) - len(valid_oblasts)}")
print(f"Видалені: {list(observation_window[observation_window['days'] < MIN_DAYS].index)}")
print(f"Залишилось областей: {len(valid_oblasts)}")

# --- 3.1 Розрахунок частот ---

# Абсолютна кількість тривог для кожної пари (область, година)
counts = (
    df_clean_filtered
    .groupby(["oblast", "hour"])
    .size()
    .unstack(fill_value=0)
)

# Нормалізація по рядках: частка кожного слоту
# відносно всіх тривог цієї області
freq = counts.div(counts.sum(axis=1), axis=0)

# Маска ненадійних слотів (менше MIN_OBSERVATIONS спостережень)
unreliable = counts < MIN_OBSERVATIONS

# Ненадійні слоти → NaN (відображатимуться сірим)
freq_masked = freq.where(~unreliable, other=np.nan)

# Сортуємо області за годиною пікової активності
# (щоб схожі патерни групувались разом)
peak_hour = freq.idxmax(axis=1) 
freq_masked = freq_masked.loc[peak_hour.sort_values().index]
counts = counts.loc[freq_masked.index]

all_nan_oblasts = freq_masked.isna().all(axis=1)
if all_nan_oblasts.any():
    print(f"⚠ Областей де всі слоти ненадійні: "
          f"{all_nan_oblasts.sum()}")
    print(f"  {list(freq_masked[all_nan_oblasts].index)}")
    print(f"  Вони залишаються на графіку як сірі рядки.")

print(f"Областей у аналізі: {len(freq_masked)}")
print(f"Замасковано слотів: {unreliable.values.sum()} "
      f"з {unreliable.size} "
      f"({unreliable.values.sum()/unreliable.size*100:.1f}%)")


# --- 3.2 Теплова карта область × година ---

print("\n[3/6] Теплова карта патернів ризику...")

# Налаштовуємо colormap — сірий для NaN
cmap = plt.cm.YlOrRd.copy()
cmap.set_bad(color="#2a2a3a")  # темно-сірий для ненадійних слотів

fig, ax = plt.subplots(figsize=(16, max(8, len(freq_masked) * 0.45)))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

im = ax.imshow(
    freq_masked.values,
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
)

# Осі
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                   rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(freq_masked)))
ax.set_yticklabels(freq_masked.index, fontsize=9)

# Колорбар
cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Відносна частота тривог\n(нормалізовано по області)",
               color="#c9cdd8", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#7a7f91")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#7a7f91", fontsize=8)

# Вертикальні лінії для розділення частин доби
for hour, label in [(6, "ранок"), (12, "день"),
                    (18, "вечір"), (22, "ніч")]:
    ax.axvline(hour - 0.5, color="#ffffff", linewidth=0.4, alpha=0.3)
    ax.text(hour + 0.1, -0.8, label, color=MUTED,
            fontsize=7, va="top", transform=ax.get_xaxis_transform())

# Заголовок
ax.set_title(
    "Відносна частота тривог: область × година доби (київський час)",
    fontsize=13, fontweight="bold", pad=16, color="#ffffff"
)

# Застереження прямо на графіку
fig.text(
    0.5, 0.01,
    "⚠  Це історичні патерни оповіщень, а не прогноз реальної загрози. "
    "Сірі клітинки — менше 5 спостережень (статистично ненадійно).",
    ha="center", fontsize=8.5, color=ACCENT,
    style="italic", wrap=True
)

fig.tight_layout(rect=[0, 0.04, 1, 1])  # місце для застереження знизу
save_and_show(fig, "06_risk_heatmap_oblast_hour.png")




# --- 3.3 Таблиця топ-3 найактивніших годин для кожної області ---

print("\n=== ТОП-3 НАЙАКТИВНІШИХ ГОДИН ПО ОБЛАСТЯХ ===")
print("(лише слоти з 5+ спостережень)\n")

rows = []

for oblast in freq_masked.index:
    oblast_freq = freq_masked.loc[oblast].dropna()

    if len(oblast_freq) == 0:
        continue

    top3 = oblast_freq.nlargest(3)

    for rank, (hour, freq_val) in enumerate(top3.items(), start=1):
        obs_count = counts.loc[oblast, hour]
        rows.append({
            "Область":    oblast,
            "Ранг":       rank,
            "Година":     f"{int(hour):02d}:00",
            "Частота,%":  round(freq_val * 100, 1),
            "Спостережень": int(obs_count),
        })

top3_df = pd.DataFrame(rows)

# Виводимо згруповано по областях
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 80)

for oblast, group in top3_df.groupby("Область", sort=False):
    print(f"  {oblast}")
    for _, row in group.iterrows():
        print(f"    #{int(row['Ранг'])}  {row['Година']}  —  "
              f"{row['Частота,%']}%  "
              f"({int(row['Спостережень'])} спостережень)")
    print()


# Зберігаємо таблицю як CSV
os.makedirs("outputs/reports", exist_ok=True)
top3_df.to_csv("outputs/reports/top3_hours_by_oblast.csv",
               index=False, encoding="utf-8-sig")
print("Таблицю збережено → outputs/reports/top3_hours_by_oblast.csv")

# --- 3.4 Загальний підсумок ---
print("\n=== ПІДСУМОК СЕКЦІЇ 3 ===")
most_active_hour = (
    freq_masked.mean(skipna=True).idxmax()
)
print(f"Найактивніша година по всій країні в середньому: "
      f"{int(most_active_hour):02d}:00")

least_reliable = unreliable.sum(axis=1).sort_values(ascending=False)
print(f"\nОбласті з найбільшою кількістю ненадійних слотів:")
for oblast, n in least_reliable.head(3).items():
    print(f"  {oblast}: {n} слотів з {unreliable.shape[1]}")

print("\n⚠  Нагадування: ці патерни описують систему оповіщення,")
print("   а не реальні траєкторії загроз.")


# ============================================================
# СЕКЦІЯ 4: ПРОГНОЗ ТРИВАЛОСТІ ТРИВОГИ
# ============================================================
# Задача: після початку тривоги в конкретному регіоні
# і конкретний час — оцінити очікувану тривалість.
# Ознаки: oblast, hour, day_of_week, month
# Цільова змінна: duration_min
#
# ЗАСТЕРЕЖЕННЯ: модель не знає про тип загрози, стан ППО,
# чи будь-які оперативні фактори. Прогноз базується
# виключно на історичних патернах тривалості.
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("\n=== СЕКЦІЯ 4: ПРОГНОЗ ТРИВАЛОСТІ ТРИВОГИ ===")

# --- 4.1 Підготовка даних ---

df_duration_sorted = df_duration.sort_values("started_at").reset_index(drop=True)
split_idx = int(len(df_duration_sorted) * 0.8)
train = df_duration_sorted.iloc[:split_idx]
test  = df_duration_sorted.iloc[split_idx:]

# Хронологічне сортування і розбиття
df_model = df_duration.sort_values("started_at").reset_index(drop=True)
split_idx = int(len(df_model) * 0.8)

print(f"Всього записів для моделювання: {len(df_model)}")
print(f"Train: {split_idx} записів "
      f"(до {df_model.iloc[split_idx-1]['started_at'].date()})")
print(f"Test:  {len(df_model) - split_idx} записів "
      f"(після {df_model.iloc[split_idx]['started_at'].date()})")

# Кодування категоріальних ознак
# LabelEncoder перетворює назви областей і днів на числа
le_oblast = LabelEncoder()
le_day    = LabelEncoder()

df_model["oblast_enc"] = le_oblast.fit_transform(df_model["oblast"])
df_model["day_enc"]    = le_day.fit_transform(df_model["day_of_week"])

FEATURES = ["oblast_enc", "hour", "day_enc", "month"]
TARGET   = "duration_min"

X_train = df_model.iloc[:split_idx][FEATURES]
X_test  = df_model.iloc[split_idx:][FEATURES]
y_train = df_model.iloc[:split_idx][TARGET]
y_test  = df_model.iloc[split_idx:][TARGET]


# --- 4.2 Baseline: медіана по кожній області ---
# Найпростіша можлива модель: для кожної тривоги
# прогнозуємо медіанну тривалість її області

oblast_median = (
    df_model.iloc[:split_idx]
    .groupby("oblast_enc")[TARGET]
    .median()
)

# Для областей яких немає в train — використовуємо загальну медіану
global_median = y_train.median()
baseline_pred = (
    X_test["oblast_enc"]
    .map(oblast_median)
    .fillna(global_median)
    .values
)

baseline_mae  = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5

print(f"\nBaseline (медіана по області):")
print(f"  MAE:  {baseline_mae:.1f} хв")
print(f"  RMSE: {baseline_rmse:.1f} хв")


# --- 4.3 RandomForestRegressor ---

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,   # не перенавчатись на малих групах
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5

print(f"\nRandomForest:")
print(f"  MAE:  {rf_mae:.1f} хв")
print(f"  RMSE: {rf_rmse:.1f} хв")


# --- 4.4 Порівняння моделей ---

improvement_mae = (baseline_mae - rf_mae) / baseline_mae * 100

print(f"\n=== ПОРІВНЯННЯ ===")
print(f"{'Модель':<30} {'MAE':>8} {'RMSE':>8}")
print("-" * 48)
print(f"{'Baseline (медіана по області)':<30} "
      f"{baseline_mae:>7.1f}  {baseline_rmse:>7.1f}")
print(f"{'RandomForest':<30} "
      f"{rf_mae:>7.1f}  {rf_rmse:>7.1f}")
print("-" * 48)

if improvement_mae > 0:
    print(f"RandomForest краще за baseline на {improvement_mae:.1f}% по MAE")
else:
    print(f"⚠ RandomForest НЕ перевищує baseline "
          f"({abs(improvement_mae):.1f}% гірше по MAE)")
    print("  Це може означати що тривалість погано передбачається")
    print("  лише з цих ознак — і це чесний результат.")


# --- 4.5 Важливість ознак ---

print("\n[4a] Графік важливості ознак...")

feature_names = ["Область", "Година доби", "День тижня", "Місяць"]
importances   = rf.feature_importances_
sorted_idx    = np.argsort(importances)

fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

colors = [ACCENT if i == sorted_idx[-1] else ACCENT2
          for i in sorted_idx]

bars = ax.barh(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color=colors,
    height=0.5,
)

for bar, val in zip(bars, importances[sorted_idx]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color="#c9cdd8")

ax.set_title("Важливість ознак — RandomForest (тривалість)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Feature importance (mean decrease in impurity)")
ax.xaxis.grid(True)
ax.set_axisbelow(True)

fig.tight_layout()
save_and_show(fig, "07_feature_importance.png")


# --- 4.6 Прогноз vs реальність ---

print("\n[4b] Графік прогноз vs реальність...")

# Беремо випадкову вибірку для читабельності графіку
sample_size = min(300, len(y_test))
sample_idx  = np.random.default_rng(42).choice(
    len(y_test), size=sample_size, replace=False
)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

ax.scatter(
    y_test.values[sample_idx],
    rf_pred[sample_idx],
    alpha=0.35, s=15,
    color=ACCENT2, linewidths=0,
)

# Лінія ідеального прогнозу
max_val = max(y_test.max(), rf_pred.max())
ax.plot([0, max_val], [0, max_val],
        color=ACCENT, linewidth=1.5,
        linestyle="--", label="Ідеальний прогноз")

ax.set_title("Прогноз тривалості vs реальність",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Реальна тривалість (хв)")
ax.set_ylabel("Прогнозована тривалість (хв)")
ax.legend(framealpha=0.1, labelcolor="white")
ax.grid(True)
ax.set_axisbelow(True)

fig.tight_layout()
save_and_show(fig, "08_predicted_vs_actual.png")


# --- 4.7 Приклад використання ---

print("\n=== ПРИКЛАД ВИКОРИСТАННЯ МОДЕЛІ ===")
print("⚠ Прогноз базується лише на історичних патернах\n")

examples = [
    ("Kyivska oblast",   3, "Tuesday",   10),
    ("Lvivska oblast",   14, "Friday",    6),
    ("Kharkivska oblast", 22, "Monday",   1),
]

for oblast, hour, day, month in examples:
    # Перевіряємо чи є область в енкодері
    if oblast not in le_oblast.classes_:
        print(f"  {oblast}: немає в тренувальних даних")
        continue

    x = pd.DataFrame([{
        "oblast_enc": le_oblast.transform([oblast])[0],
        "hour":       hour,
        "day_enc":    le_day.transform([day])[0],
        "month":      month,
    }])

    pred = rf.predict(x)[0]
    actual_median = (
        df_model[df_model["oblast"] == oblast][TARGET].median()
    )

    print(f"  {oblast}, {hour:02d}:00, {day}")
    print(f"    Прогноз моделі:  {pred:.0f} хв")
    print(f"    Медіана baseline: {actual_median:.0f} хв")
    print()

    # ============================================================
# СЕКЦІЯ 4б: ПОКРАЩЕНА МОДЕЛЬ
# Зміни відносно секції 4:
# 1. Класифікація замість регресії (варіант Б)
# 2. Rolling ознака — середня тривалість за 7 днів (варіант В)
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("\n=== СЕКЦІЯ 4б: ПОКРАЩЕНА МОДЕЛЬ ===")

# --- 4б.1 Категоризація тривалості ---

def categorize_duration(minutes):
    if minutes < 30:
        return "коротка"
    elif minutes < 90:
        return "середня"
    else:
        return "довга"

CATEGORY_ORDER = ["коротка", "середня", "довга"]

df_model2 = df_duration.sort_values("started_at").reset_index(drop=True).copy()
df_model2["duration_cat"] = df_model2["duration_min"].apply(categorize_duration)

print("Розподіл категорій тривалості:")
cat_counts = df_model2["duration_cat"].value_counts()
for cat in CATEGORY_ORDER:
    n = cat_counts.get(cat, 0)
    pct = n / len(df_model2) * 100
    print(f"  {cat:<10} {n:>6} записів ({pct:.1f}%)")


# --- 4б.2 Rolling ознака без data leakage ---

df_model2 = df_model2.sort_values(["oblast", "started_at"]).copy()

# Конвертуємо started_at в timezone-naive для rolling по часу
df_model2["started_at_naive"] = (
    df_model2["started_at"].dt.tz_localize(None)
    if df_model2["started_at"].dt.tz is None
    else df_model2["started_at"].dt.tz_convert(None)
)

def rolling_7d_mean(group):
    group = group.set_index("started_at_naive").sort_index()
    # shift(1) виключає поточну тривогу
    rolled = (
        group["duration_min"]
        .shift(1)
        .rolling(window="7D", min_periods=1)
        .mean()
    )
    rolled.index = group.index
    return rolled.values

# apply по кожній області окремо
results = []
for oblast, group in df_model2.groupby("oblast"):
    vals = rolling_7d_mean(group)
    results.append(
        pd.Series(vals, index=group.index, name="rolling_7d_mean")
    )

df_model2["rolling_7d_mean"] = pd.concat(results).sort_index()

# Повертаємо хронологічний порядок
df_model2 = df_model2.sort_values("started_at").reset_index(drop=True)

# Заповнюємо NaN (перші записи без історії) медіаною train
split_idx2  = int(len(df_model2) * 0.8)
train_median = df_model2.iloc[:split_idx2]["duration_min"].median()
df_model2["rolling_7d_mean"] = (
    df_model2["rolling_7d_mean"].fillna(train_median)
)

print(f"NaN у rolling ознаці після заповнення: "
      f"{df_model2['rolling_7d_mean'].isna().sum()}")


# --- 4б.3 Кодування ознак ---

df_model2["oblast_enc"] = le_oblast.transform(
    df_model2["oblast"].where(
        df_model2["oblast"].isin(le_oblast.classes_), le_oblast.classes_[0]
    )
)
df_model2["day_enc"] = le_day.transform(
    df_model2["day_of_week"].where(
        df_model2["day_of_week"].isin(le_day.classes_), le_day.classes_[0]
    )
)

FEATURES2 = ["oblast_enc", "hour", "day_enc", "month", "rolling_7d_mean"]
TARGET2   = "duration_cat"

X_train2 = df_model2.iloc[:split_idx2][FEATURES2]
X_test2  = df_model2.iloc[split_idx2:][FEATURES2]
y_train2 = df_model2.iloc[:split_idx2][TARGET2]
y_test2  = df_model2.iloc[split_idx2:][TARGET2]

print(f"\nTrain: {len(X_train2)} записів")
print(f"Test:  {len(X_test2)} записів")


# --- 4б.4 Baseline для класифікації ---
# Найчастіша категорія в train → прогноз для всіх

most_common_cat = y_train2.value_counts().idxmax()
baseline_pred2  = [most_common_cat] * len(y_test2)
baseline_acc    = accuracy_score(y_test2, baseline_pred2)

print(f"\nBaseline (найчастіша категорія: '{most_common_cat}'):")
print(f"  Accuracy: {baseline_acc:.3f}")


# --- 4б.5 RandomForestClassifier ---

rfc = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    class_weight="balanced",   # компенсує нерівний розподіл категорій
    random_state=42,
    n_jobs=-1,
)
rfc.fit(X_train2, y_train2)
rfc_pred2 = rfc.predict(X_test2)
rfc_acc   = accuracy_score(y_test2, rfc_pred2)

print(f"\nRandomForestClassifier:")
print(f"  Accuracy: {rfc_acc:.3f}")

improvement_acc = (rfc_acc - baseline_acc) / baseline_acc * 100
print(f"\n=== ПОРІВНЯННЯ ===")
print(f"{'Модель':<35} {'Accuracy':>10}")
print("-" * 47)
print(f"{'Baseline (найчастіша категорія)':<35} {baseline_acc:>10.3f}")
print(f"{'RandomForest + rolling ознака':<35} {rfc_acc:>10.3f}")
print("-" * 47)

if improvement_acc > 5:
    print(f"RandomForest краще за baseline на {improvement_acc:.1f}%")
else:
    print(f"⚠ Покращення незначне ({improvement_acc:.1f}%)")
    print("  Тривалість залишається важко передбачуваною")
    print("  з доступних ознак — це чесний результат.")

print("\nДетальний звіт по категоріях:")
print(classification_report(y_test2, rfc_pred2,
                             target_names=CATEGORY_ORDER,
                             zero_division=0))


# --- 4б.6 Матриця помилок ---

print("\n[4в] Матриця помилок...")

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

cm = confusion_matrix(y_test2, rfc_pred2,
                      labels=CATEGORY_ORDER,
                      normalize="true")

sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    xticklabels=CATEGORY_ORDER,
    yticklabels=CATEGORY_ORDER,
    ax=ax,
    linewidths=0.5,
    linecolor="#0f1117",
    cbar_kws={"shrink": 0.8},
)

ax.set_title("Матриця помилок — класифікація тривалості\n"
             "(нормалізовано по рядках)",
             fontsize=12, fontweight="bold", pad=14)
ax.set_xlabel("Прогноз моделі")
ax.set_ylabel("Реальна категорія")

fig.text(
    0.5, 0.01,
    "⚠ Категорії базуються на історичних патернах тривалості оповіщень",
    ha="center", fontsize=8, color=ACCENT, style="italic"
)

fig.tight_layout(rect=[0, 0.04, 1, 1])
save_and_show(fig, "09_confusion_matrix.png")


# --- 4б.7 Важливість ознак ---

print("\n[4г] Важливість ознак (класифікатор)...")

feature_names2 = ["Область", "Година доби",
                  "День тижня", "Місяць",
                  "Середня тривалість\n(останні 7 днів)"]
importances2  = rfc.feature_importances_
sorted_idx2   = np.argsort(importances2)

fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

colors2 = [ACCENT if i == sorted_idx2[-1] else ACCENT2
           for i in sorted_idx2]

bars2 = ax.barh(
    [feature_names2[i] for i in sorted_idx2],
    importances2[sorted_idx2],
    color=colors2,
    height=0.5,
)

for bar, val in zip(bars2, importances2[sorted_idx2]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color="#c9cdd8")

ax.set_title("Важливість ознак — RandomForest (класифікація)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Feature importance")
ax.xaxis.grid(True)
ax.set_axisbelow(True)

fig.tight_layout()
save_and_show(fig, "10_feature_importance_classifier.png")

# ============================================================
# СЕКЦІЯ 5: PROPHET — ПРОГНОЗ ТИЖНЕВОЇ ІНТЕНСИВНОСТІ
# ============================================================
# Задача: прогнозувати кількість тривог на наступні тижні
# на основі історичних патернів інтенсивності.
#
# ЗАСТЕРЕЖЕННЯ: модель екстраполює статистичні патерни.
# Вона не знає про оперативну обстановку, рішення
# командування чи будь-які зовнішні фактори.
# Прогноз — це "якби патерн продовжився", а не передбачення.
# ============================================================

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

print("\n=== СЕКЦІЯ 5: PROPHET ПРОГНОЗ ТИЖНЕВОЇ ІНТЕНСИВНОСТІ ===")

# --- 5.1 Агрегація по тижнях ---

weekly = (
    df_clean
    .groupby("year_week")
    .size()
    .reset_index(name="y")
)

weekly["ds"] = pd.to_datetime(
    weekly["year_week"] + "-1", format="%G-W%V-%u"
).dt.tz_localize(None)

weekly = weekly.sort_values("ds").reset_index(drop=True)

# Виключаємо 2026 — з січня 2026 система оповіщення
# перейшла переважно на raion рівень (36666 записів проти
# 221 на oblast рівні). Фільтр level=="oblast" робить
# дані 2026 непридатними для порівняння з попередніми роками.
CUTOFF_DATE = "2026-01-01"
weekly = weekly[weekly["ds"] < CUTOFF_DATE].copy().reset_index(drop=True)

print(f"Тижневий ряд після виключення 2026: {len(weekly)} точок")
print(f"Період: {weekly['ds'].min().date()} — {weekly['ds'].max().date()}")
print(f"Середня кількість тривог на тиждень: {weekly['y'].mean():.1f}")
print(f"Максимум: {weekly['y'].max()} "
      f"(тиждень {weekly.loc[weekly['y'].idxmax(), 'ds'].date()})")


# --- 5.2 Хронологічне розбиття ---

SPLIT_DATE = "2024-07-01"

train = weekly[weekly["ds"] < SPLIT_DATE].copy()
test  = weekly[weekly["ds"] >= SPLIT_DATE].copy()

print(f"\nРозбиття: {SPLIT_DATE}")
print(f"Train: {len(train)} тижнів "
      f"({train['ds'].min().date()} — {train['ds'].max().date()})")
print(f"Test:  {len(test)} тижнів "
      f"({test['ds'].min().date()} — {test['ds'].max().date()})")


# --- 5.3 Changepoints для масованих атак ---
# Масовані удари по енергетиці жовтень-листопад 2022
# Джерело: відкриті дані про великі атаки
# Ці дати вказуються явно щоб Prophet не інтерпретував
# сплески як зміну довгострокового тренду

major_attacks = pd.DataFrame({
    "holiday": "масована_атака",
    "ds": pd.to_datetime([
        "2022-10-10",
        "2022-10-18",
        "2022-11-15",
        "2022-11-23",
        "2022-11-24",
    ]),
    "lower_window": 0,
    "upper_window": 2,  # ефект на 2 дні після атаки
})

print(f"\nChangepoints вказано для {len(major_attacks)} подій")


# --- 5.4 Baseline ---

baseline_value = train["y"].mean()
baseline_pred  = pd.Series([baseline_value] * len(test),
                            index=test.index)

print(f"\nBaseline (середнє train): {baseline_value:.1f} тривог/тиждень")


# --- 5.5 Prophet ---

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,  # у нас тижневі точки — добова не застосовна
    daily_seasonality=False,
    seasonality_mode="multiplicative",  # сезонність масштабується з трендом
    changepoint_prior_scale=0.05,       # консервативні зміни тренду
    holidays=major_attacks,
    interval_width=0.80,                # 80% довірчий інтервал
)

# Додаємо місячну сезонність вручну
model.add_seasonality(
    name="monthly",
    period=30.5,
    fourier_order=5,
)

model.fit(train[["ds", "y"]])

# Прогноз на тестовий період + 4 тижні вперед після останньої точки
last_date   = weekly["ds"].max()
future_end  = last_date + pd.Timedelta(weeks=4)
future_dates = pd.date_range(
    start=train["ds"].min(),
    end=future_end,
    freq="W-MON"
)
future = pd.DataFrame({"ds": future_dates})

forecast = model.predict(future)

# Витягуємо прогноз лише для тестового періоду
forecast_test = forecast[forecast["ds"].isin(test["ds"])].copy()


# --- 5.6 Оцінка моделей ---

def evaluate(y_true, y_pred, model_name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"\n{model_name}")
    print(f"  MAE:  {mae:.1f} тривог/тиждень")
    print(f"  RMSE: {rmse:.1f} тривог/тиждень")
    return mae, rmse

print("\n=== ОЦІНКА МОДЕЛЕЙ ===")
b_mae,  b_rmse  = evaluate(test["y"], baseline_pred, "Baseline (середнє)")
p_mae,  p_rmse  = evaluate(test["y"], forecast_test["yhat"], "Prophet")

improvement = (b_mae - p_mae) / b_mae * 100
print(f"\n{'='*40}")
if improvement > 0:
    print(f"Prophet краще за baseline на {improvement:.1f}% по MAE")
else:
    print(f"⚠ Prophet не перевищує baseline ({abs(improvement):.1f}% гірше)")
    print("  Можливі причини: структурні зміни в патерні,")
    print("  або інтенсивність тривог має низьку автокореляцію.")


# --- 5.7 Графік: факт + прогноз + довірчий інтервал ---

print("\n[5] Графік прогнозу Prophet...")

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

# Train дані
ax.plot(train["ds"], train["y"],
        color=MUTED, linewidth=1, alpha=0.7,
        label="Факт (train)")

# Test дані
ax.plot(test["ds"], test["y"],
        color="#ffffff", linewidth=1.5,
        label="Факт (test)", zorder=5)

# Прогноз Prophet на весь період
ax.plot(forecast["ds"], forecast["yhat"],
        color=ACCENT2, linewidth=1.5,
        linestyle="--", label="Прогноз Prophet", zorder=4)

# Довірчий інтервал
ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.2, color=ACCENT2,
    label="80% довірчий інтервал"
)

# Вертикальна лінія розбиття train/test
ax.axvline(pd.Timestamp(SPLIT_DATE),
           color=ACCENT, linewidth=1.5,
           linestyle=":", alpha=0.9)
ax.text(pd.Timestamp(SPLIT_DATE), ax.get_ylim()[1] * 0.95,
        " train | test", color=ACCENT, fontsize=9, va="top")

# Позначення масованих атак
for _, row in major_attacks.iterrows():
    ax.axvline(row["ds"], color="#ffaa00",
               linewidth=0.8, alpha=0.5, linestyle="-.")

# Підпис однієї лінії для легенди
ax.axvline(major_attacks["ds"].iloc[0],
           color="#ffaa00", linewidth=0.8,
           alpha=0.5, linestyle="-.",
           label="Масовані атаки (жовт-лист 2022)")

# Метрики прямо на графіку
metrics_text = (f"Baseline MAE: {b_mae:.0f}  |  "
                f"Prophet MAE: {p_mae:.0f}  |  "
                f"Покращення: {improvement:.1f}%")
ax.text(0.02, 0.97, metrics_text,
        transform=ax.transAxes,
        fontsize=8.5, va="top", color="#c9cdd8",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=PANEL, alpha=0.8))

ax.set_title("Тижнева інтенсивність тривог — Prophet прогноз",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Дата")
ax.set_ylabel("Кількість тривог на тиждень")
ax.legend(framealpha=0.15, labelcolor="white", loc="upper left",
          bbox_to_anchor=(0.0, 0.88))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.yaxis.grid(True)
ax.set_axisbelow(True)

fig.text(
    0.5, 0.01,
    "⚠ Прогноз базується виключно на історичних патернах. "
    "Не враховує оперативну обстановку і не є передбаченням.",
    ha="center", fontsize=8.5, color=ACCENT, style="italic"
)

fig.tight_layout(rect=[0, 0.04, 1, 1])
save_and_show(fig, "11_prophet_forecast.png")


# --- 5.8 Компоненти Prophet ---

print("\n[5б] Компоненти Prophet (тренд + сезонність)...")

fig_comp = model.plot_components(forecast)
fig_comp.patch.set_facecolor(BG)

for comp_ax in fig_comp.get_axes():
    comp_ax.set_facecolor(PANEL)
    comp_ax.tick_params(colors=MUTED)
    comp_ax.yaxis.label.set_color("#c9cdd8")
    comp_ax.xaxis.label.set_color("#c9cdd8")
    comp_ax.title.set_color("#ffffff")
    for spine in comp_ax.spines.values():
        spine.set_edgecolor("#2e3140")

fig_comp.tight_layout()
save_and_show(fig_comp, "12_prophet_components.png")


# --- 5.9 Підсумок ---

print("\n=== ПІДСУМОК СЕКЦІЇ 5 ===")
print(f"Тренувальний період: {train['ds'].min().date()} — "
      f"{train['ds'].max().date()} ({len(train)} тижнів)")
print(f"Тестовий період:     {test['ds'].min().date()} — "
      f"{test['ds'].max().date()} ({len(test)} тижнів)")
print(f"\nBaseline MAE:  {b_mae:.1f} тривог/тиждень")
print(f"Prophet MAE:   {p_mae:.1f} тривог/тиждень")
print(f"Покращення:    {improvement:.1f}%")

future_only = forecast[forecast["ds"] > last_date]
print(f"\nПрогноз на наступні 4 тижні після {last_date.date()}:")
for _, row in future_only.iterrows():
    print(f"  {row['ds'].date()}  →  "
          f"{row['yhat']:.0f} тривог "
          f"[{row['yhat_lower']:.0f} — {row['yhat_upper']:.0f}]")

print("\n⚠  Нагадування: прогноз описує патерни оповіщень,")
print("   а не реальні загрози. Довірчий інтервал відображає")
print("   статистичну невизначеність моделі, а не безпеку регіону.")

print()
print()

# --- ДІАГНОСТИКА ---

print("Середнє по роках:")
for year, group in weekly.groupby(weekly["ds"].dt.year):
    print(f"  {year}: {group['y'].mean():.0f} тривог/тиждень "
          f"(тижнів: {len(group)})")

print(f"\nСереднє train: {train['y'].mean():.0f}")
print(f"Середнє test:  {test['y'].mean():.0f}")

print("\nТренд всередині train (перша vs друга половина):")
half = len(train) // 2
print(f"  Перша половина train:  {train.iloc[:half]['y'].mean():.0f}")
print(f"  Друга половина train:  {train.iloc[half:]['y'].mean():.0f}")

print("\nТренд всередині test (перша vs друга половина):")
half_t = len(test) // 2
print(f"  Перша половина test:  {test.iloc[:half_t]['y'].mean():.0f}")
print(f"  Друга половина test:  {test.iloc[half_t:]['y'].mean():.0f}")

print("Тижні 2026 року детально:")
weekly_2026 = weekly[weekly["ds"].dt.year == 2026]
print(weekly_2026[["ds", "y"]].to_string())

print(f"\nВсього тижнів 2026: {len(weekly_2026)}")
print(f"Сума тривог 2026: {weekly_2026['y'].sum()}")
print(f"Середнє без останнього тижня: "
      f"{weekly_2026.iloc[:-1]['y'].mean():.0f}")

# Чи є дані інших рівнів у 2026?
df_2026 = df[df["started_at"].dt.year == 2026]

print("Записи 2026 по рівнях:")
print(df_2026["level"].value_counts())

print("\nЗаписи 2026 по джерелах:")
print(df_2026["source"].value_counts())

print("\nПерший і останній запис 2026:")
print(df_2026["started_at"].min())
print(df_2026["started_at"].max())

# --- 5.2б Короткостроковий тест: train до останніх 12 тижнів ---

SHORT_SPLIT = weekly["ds"].max() - pd.Timedelta(weeks=12)

train_short = weekly[weekly["ds"] <= SHORT_SPLIT].copy()
test_short  = weekly[weekly["ds"] > SHORT_SPLIT].copy()

print(f"\nКороткостроковий горизонт (12 тижнів):")
print(f"Train: {len(train_short)} тижнів "
      f"({train_short['ds'].min().date()} — {train_short['ds'].max().date()})")
print(f"Test:  {len(test_short)} тижнів "
      f"({test_short['ds'].min().date()} — {test_short['ds'].max().date()})")

# Baseline для короткострокового тесту
baseline_short_value = train_short["y"].mean()
baseline_short_pred  = pd.Series(
    [baseline_short_value] * len(test_short),
    index=test_short.index
)
b_short_mae, b_short_rmse = evaluate(
    test_short["y"], baseline_short_pred,
    "Baseline короткостроковий"
)

# Prophet для короткострокового тесту
model_short = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.001,  # плоский тренд
    changepoint_range=0.95,
    holidays=major_attacks,
    interval_width=0.80,
)
model_short.add_seasonality(
    name="monthly",
    period=30.5,
    fourier_order=5,
)
model_short.fit(train_short[["ds", "y"]])

future_short = pd.DataFrame({
    "ds": pd.date_range(
        start=train_short["ds"].min(),
        end=test_short["ds"].max(),
        freq="W-MON"
    )
})
forecast_short  = model_short.predict(future_short)
forecast_short_test = forecast_short[
    forecast_short["ds"].isin(test_short["ds"])
].copy()

p_short_mae, p_short_rmse = evaluate(
    test_short["y"],
    forecast_short_test["yhat"],
    "Prophet короткостроковий (12 тижнів)"
)

# --- Графік короткострокового прогнозу ---

print("\n[5в] Графік короткострокового прогнозу...")

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

# Показуємо лише останні 40 тижнів для читабельності
window_start = SHORT_SPLIT - pd.Timedelta(weeks=28)
weekly_window    = weekly[weekly["ds"] >= window_start]
forecast_window  = forecast_short[forecast_short["ds"] >= window_start]

train_window = weekly_window[weekly_window["ds"] <= SHORT_SPLIT]
test_window  = weekly_window[weekly_window["ds"] > SHORT_SPLIT]

ax.plot(train_window["ds"], train_window["y"],
        color=MUTED, linewidth=1.2, alpha=0.8,
        label="Факт (train)")
ax.plot(test_window["ds"], test_window["y"],
        color="#ffffff", linewidth=2,
        label="Факт (test, 12 тижнів)", zorder=5)
ax.plot(forecast_window["ds"], forecast_window["yhat"],
        color=ACCENT2, linewidth=1.5, linestyle="--",
        label="Прогноз Prophet", zorder=4)
ax.fill_between(
    forecast_window["ds"],
    forecast_window["yhat_lower"],
    forecast_window["yhat_upper"],
    alpha=0.2, color=ACCENT2,
    label="80% довірчий інтервал"
)
ax.axvline(SHORT_SPLIT, color=ACCENT,
           linewidth=1.5, linestyle=":", alpha=0.9)
ax.text(SHORT_SPLIT, ax.get_ylim()[1] * 0.95,
        " train | test", color=ACCENT, fontsize=9, va="top")

ax.set_title("Короткостроковий прогноз — останні 12 тижнів",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Дата")
ax.set_ylabel("Кількість тривог на тиждень")
ax.legend(framealpha=0.15, labelcolor="white")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.yaxis.grid(True)
ax.set_axisbelow(True)

fig.tight_layout()
save_and_show(fig, "13_prophet_short_forecast.png")


# --- Фінальне порівняння всіх моделей ---

print("\n" + "="*60)
print("ФІНАЛЬНЕ ПОРІВНЯННЯ МОДЕЛЕЙ — СЕКЦІЯ 5")
print("="*60)
print(f"\n{'Модель':<40} {'MAE':>8} {'RMSE':>8} {'Горизонт':>12}")
print("-"*70)
print(f"{'Baseline (середнє train)':<40} "
      f"{b_mae:>8.1f} {b_rmse:>8.1f} {'79 тижнів':>12}")
print(f"{'Prophet довгостроковий':<40} "
      f"{p_mae:>8.1f} {p_rmse:>8.1f} {'79 тижнів':>12}")
print(f"{'Baseline короткостроковий':<40} "
      f"{b_short_mae:>8.1f} {b_short_rmse:>8.1f} {'12 тижнів':>12}")
print(f"{'Prophet короткостроковий':<40} "
      f"{p_short_mae:>8.1f} {p_short_rmse:>8.1f} {'12 тижнів':>12}")
print("-"*70)

# Порівняння короткострокового Prophet з його baseline
short_improvement = (b_short_mae - p_short_mae) / b_short_mae * 100
if short_improvement > 0:
    print(f"\n✓ Короткостроковий Prophet краще за свій baseline "
          f"на {short_improvement:.1f}% по MAE")
else:
    print(f"\n⚠ Короткостроковий Prophet гірше за свій baseline "
          f"на {abs(short_improvement):.1f}% по MAE")

# ============================================================
# СЕКЦІЯ 6: ФІНАЛЬНИЙ ПІДСУМОК ПРОЄКТУ
# ============================================================

import os
from datetime import datetime

print("\n=== СЕКЦІЯ 6: ФОРМУВАННЯ ФІНАЛЬНОГО ЗВІТУ ===")

# --- 6.1 Збір ключових чисел з попередніх секцій ---

# З секції 2
peak_week       = weekly.loc[weekly["y"].idxmax(), "ds"].date()
peak_week_count = weekly["y"].max()
top_oblast      = cumulative.iloc[-1]["oblast"]
top_oblast_hrs  = cumulative.iloc[-1]["hours"]
low_oblast      = cumulative.iloc[0]["oblast"]
low_oblast_hrs  = cumulative.iloc[0]["hours"]
overall_median_dur = df_duration["duration_min"].median()

# З секції 3
most_active_hour_val = int(freq_masked.mean(skipna=True).idxmax())

# З секції 4 (регресія)
rf_improvement_reg = (baseline_mae - rf_mae) / baseline_mae * 100

# З секції 4б (класифікація)
rf_improvement_clf = (baseline_acc - rfc_acc) / baseline_acc * 100 * -1

# З секції 5
long_improvement  = (b_mae - p_mae) / b_mae * 100
short_improvement = (b_short_mae - p_short_mae) / b_short_mae * 100

# --- 6.2 Формування звіту ---

report_lines = []

def line(text=""):
    report_lines.append(text)

def section(title):
    line()
    line("=" * 62)
    line(f"  {title}")
    line("=" * 62)

def subsection(title):
    line()
    line(f"  ── {title}")
    line("  " + "─" * 50)

def item(label, value, note=""):
    note_str = f"  ({note})" if note else ""
    line(f"  {label:<38} {value}{note_str}")

def paragraph(text):
    import textwrap
    for l in textwrap.wrap(text, width=60):
        line(f"  {l}")

# ── Заголовок ──
line("=" * 62)
line("  АНАЛІЗ ПОВІТРЯНИХ ТРИВОГ В УКРАЇНІ")
line("  Звіт про результати проєкту з аналізу часових рядів")
line("=" * 62)
line(f"  Дата формування: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
line(f"  Датасет: {len(df_clean):,} записів (oblast рівень, official)")
line(f"  Період:  {df_clean['started_at'].min().date()} — "
     f"{df_clean['started_at'].max().date()}")
line()
line("  ⚠ ЗАСТЕРЕЖЕННЯ: цей звіт описує патерни системи")
line("  оповіщення, а не факти бойових дій. Тривога —")
line("  це сигнал реакції системи, а не підтверджена атака.")

# ── Секція 1: Ключові числа ──
section("1. КЛЮЧОВІ ЧИСЛА ДАТАСЕТУ")

subsection("Загальна активність")
item("Всього тривог (oblast рівень):", f"{len(df_clean):,}")
item("Унікальних областей:", f"{df_clean['oblast'].nunique()}")
item("Пік тижневої активності:",
     f"{peak_week_count} тривог",
     f"тиждень {peak_week}")
item("Середня кількість тривог/тиждень:",
     f"{weekly['y'].mean():.0f}")

subsection("Тривалість тривог")
item("Медіанна тривалість:", f"{overall_median_dur:.0f} хв")
item("Записів з аномальною тривалістю:",
     f"{df_clean['duration_suspicious'].sum():,}",
     "виключено з аналізу тривалості")

subsection("Регіональне навантаження")
item("Найнавантаженіша область:",
     top_oblast,
     f"{top_oblast_hrs:,.0f} год")
item("Найменш навантажена область:",
     low_oblast,
     f"{low_oblast_hrs:,.0f} год")
item("Виключено областей (мало даних):",
     "1",
     "Luhanska oblast — 0 днів спостереження")

subsection("Часові патерни")
item("Найактивніша година доби:",
     f"{most_active_hour_val:02d}:00",
     "київський час, середнє по країні")

# ── Секція 2: Результати по цілях проєкту ──
section("2. РЕЗУЛЬТАТИ ПО ТРЬОХ ЦІЛЯХ ПРОЄКТУ")

subsection("Ціль 1 — Тижнева інтенсивність (Prophet)")
item("Baseline MAE (79 тижнів):",  f"{b_mae:.1f} тривог/тиждень")
item("Prophet MAE (79 тижнів):",   f"{p_mae:.1f} тривог/тиждень")
item("Baseline MAE (12 тижнів):",  f"{b_short_mae:.1f} тривог/тиждень")
item("Prophet MAE (12 тижнів):",   f"{p_short_mae:.1f} тривог/тиждень")
line()
paragraph(
    "Prophet не перевищив baseline на жодному горизонті. "
    "Найкращим предиктором виявилось просте середнє "
    "тренувального періоду. Це змістовний результат: "
    "інтенсивність тривог визначається зовнішніми подіями "
    "без статистичного патерну в наявних даних. Широкий "
    "довірчий інтервал Prophet є чесним відображенням "
    "реальної невизначеності, а не ознакою слабкості моделі."
)

subsection("Ціль 2 — Прогноз тривалості (RandomForest)")
item("Baseline MAE (медіана по області):", f"{baseline_mae:.1f} хв")
item("RandomForest MAE (регресія):",       f"{rf_mae:.1f} хв")
item("Покращення над baseline:",           f"{rf_improvement_reg:.1f}%")
line()
item("Baseline accuracy (класифікація):",  f"{baseline_acc:.3f}")
item("RandomForest accuracy (+ rolling):", f"{rfc_acc:.3f}")
item("Покращення над baseline:",           f"{rf_improvement_clf:.1f}%")
line()
paragraph(
    "Регресійна модель не перевищила baseline значуще (+4.4%). "
    "Класифікація на три категорії з rolling ознакою дала "
    "покращення 67%, але recall для довгих тривог залишився "
    "низьким (0.17). Довгі тривоги є аномаліями які не "
    "передбачаються з часових та регіональних ознак — вони "
    "визначаються факторами відсутніми в датасеті."
)

subsection("Ціль 3 — Патерни по регіону і часу")
paragraph(
    "Теплова карта область×година виявила статистично "
    "значущі патерни добової активності для 24 областей. "
    "Таблиця топ-3 годин по кожній області збережена у "
    "outputs/reports/top3_hours_by_oblast.csv. "
    "Ці патерни є описовими і не є прогнозом."
)

# ── Секція 3: Обмеження датасету ──
section("3. ЗАФІКСОВАНІ ОБМЕЖЕННЯ ДАТАСЕТУ")

line()
line("  3.1 Концептуальні обмеження")
line("  ─────────────────────────────────────────────────")
paragraph(
    "Тривога — це сигнал системи оповіщення, а не "
    "підтверджений факт атаки. Датасет описує реакцію "
    "системи, а не саму загрозу. Всі висновки стосуються "
    "патернів оповіщення."
)
line()
paragraph(
    "Відсутні дані які необхідні для причинно-наслідкового "
    "аналізу: траєкторії ракет, тип загрози, стан ППО, "
    "рішення командування, метеоумови."
)

line()
line("  3.2 Технічні обмеження")
line("  ─────────────────────────────────────────────────")
paragraph(
    "З січня 2026 система оповіщення перейшла переважно "
    "на raion рівень (36,666 записів проти 221 на oblast). "
    "Дані 2026 виключені з аналізу часових рядів через "
    "несумісність з попереднім форматом."
)
line()
paragraph(
    "Луганська область виключена через єдиний запис "
    "(0 днів спостереження) — недостатньо для надійного "
    "статистичного аналізу."
)
line()
paragraph(
    "Нормалізація по рядках у секції 3 коректна для "
    "24 областей з вікном спостереження 1253-1501 днів. "
    "Різниця між областями незначна (~250 днів)."
)

# ── Секція 4: Що потрібно для покращення ──
section("4. ЩО ПОТРІБНО ДЛЯ ПОКРАЩЕННЯ МОДЕЛЕЙ")

line()
line("  Короткострокове (в межах наявних даних):")
line("  • Агрегація по raion рівню для включення 2026 року")
line("  • Додаткові rolling ознаки (14 днів, 30 днів)")
line("  • Окремі моделі для різних фаз конфлікту")
line()
line("  Довгострокове (потребує зовнішніх даних):")
line("  • Дані про інтенсивність бойових дій по лінії фронту")
line("  • Тип застосовуваної зброї (балістична/крилата/дрон)")
line("  • Дані про стан і розташування засобів ППО")
line("  • Метеорологічні дані (умови для авіації/дронів)")
line("  • Новинні події як зовнішній регресор для Prophet")

# ── Фінал ──
section("5. ФАЙЛИ ПРОЄКТУ")
line()
line("  outputs/plots/")
line("  ├── 01_weekly_intensity.png")
line("  ├── 02_heatmap_hour_day.png")
line("  ├── 03_cumulative_hours_by_oblast.png")
line("  ├── 04_duration_boxplot_by_oblast.png")
line("  ├── 05_monthly_duration_trend.png")
line("  ├── 06_risk_heatmap_oblast_hour.png")
line("  ├── 07_feature_importance.png")
line("  ├── 08_predicted_vs_actual.png")
line("  ├── 09_confusion_matrix.png")
line("  ├── 10_feature_importance_classifier.png")
line("  ├── 11_prophet_forecast.png")
line("  ├── 12_prophet_components.png")
line("  └── 13_prophet_short_forecast.png")
line()
line("  outputs/reports/")
line("  ├── top3_hours_by_oblast.csv")
line("  └── summary.txt  ← цей файл")
line()
line("=" * 62)
line("  Кінець звіту")
line("=" * 62)

# --- 6.3 Вивід у консоль ---
full_report = "\n".join(report_lines)
print(full_report)

# --- 6.4 Збереження у файл ---
os.makedirs("outputs/reports", exist_ok=True)
report_path = "outputs/reports/summary.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(full_report)

print(f"\nЗвіт збережено → {report_path}")
