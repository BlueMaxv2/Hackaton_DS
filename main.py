import datetime
import pandas as pd
import numpy as np

def process_dataframe(file_path, column_names):
    print(f"\n--- Обработка файла: {file_path} ---")

    # Загрузка данных
    df = pd.read_csv(
        file_path,
        sep="^",
        encoding="cp1251",
        on_bad_lines="skip",
        header=None,
        decimal=','
    )

    # Присваиваем новые названия колонкам
    df.columns = column_names
    print(f"Исходное количество строк: {len(df)}")

    # Очистка лишних пробелов в строковых данных
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Преобразование пустых строк в NaN
    initial_total_nan = df.isnull().sum().sum()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace('', np.nan)
    final_total_nan = df.isnull().sum().sum()

    # Удаление дубликатов
    df.drop_duplicates(inplace=True)

    # Удаление строк с хотя бы одним пропущенным значением (NaN)
    rows_before_na_drop = len(df)
    df.dropna(inplace=True)
    print(f"Итоговое количество строк: {len(df)}")

    # Приведение столбцов к стандартным названиям (если есть пробелы и лишние символы)
    df.columns = df.columns.str.strip()

    # Приведение типов: даты
    # Список колонок с датами для текущего DataFrame
    date_cols = [
        "Дата записи среза", # Для "Срезы2024_2025.csv"
        "Дата и время начала операции",
        "Дата и время окончания операции",
        "Дата закрытия операции" # Для "ОтметкаФакта.csv"
    ]
    for col in date_cols:
        if col in df.columns:
            # Важно: используем errors="coerce", чтобы не упасть, если формат не подходит
            df[col] = pd.to_datetime(df[col], format="%d.%m.%Y %H:%M:%S", errors="coerce")
            # После преобразования в datetime, если остаются NaT, то удаляем
            df.dropna(subset=[col], inplace=True)
        # else: # Закомментировано, чтобы не выводить предупреждения для колонок, которых нет в данном DF
            # print(f"Предупреждение: Колонка '{col}' для дат не найдена.")

    # Приведение типов: числовые
    numeric_cols_float = [
        "Общая трудоемкость",
        "Остаточная трудоемкость",
        "Закрытая трудоемкость" # Для "ОтметкаФакта.csv"
    ]

    numeric_cols_int = [
        "Количество деталей",
        "Количество рабочих на операцию"
    ]

    for col in numeric_cols_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            # После преобразования в numeric, если остаются NaN, то удаляем
            df.dropna(subset=[col], inplace=True)
        # else:
            # print(f"Предупреждение: Колонка '{col}' (ожидался float) не найдена.")

    for col in numeric_cols_int:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype('Int64')
            # После преобразования в numeric, если остаются NaN, то удаляем
            df.dropna(subset=[col], inplace=True)
        # else:
            # print(f"Предупреждение: Колонка '{col}' (ожидался Int64) не найдена.")

    print(f"\nИнформация о данных после предобработки для {file_path}:")
    print(df.info())

    print(f"\nПервые строки таблицы после предобработки для {file_path}:")
    print(df.head())

    return df

# --- Основная часть скрипта ---

# 1. Данные для первой таблицы "Срезы2024_2025.csv"
srez_file = "Срезы2024_2025.csv"
srez_column_names = [
    "Дата записи среза",
    "Номер заказа на производство",
    "Артикул конечного изделия",
    "Наименование конечного изделия",
    "Наименование станции",
    "Артикул номенклатуры",
    "Наименование номенклатуры",
    "Количество деталей",
    "Номер операции",
    "Название операции",
    "Общая трудоемкость",
    "Остаточная трудоемкость",
    "Количество рабочих на операцию",
    "Подразделение выполнения операции плановое",
    "ШГО – вид оборудования",
    "Рабочий станок",
    "Артикул узла вовлечения",
    "Наименование узла вовлечения",
    "Признак закрытия операции",
    "Дата и время начала операции",
    "Дата и время окончания операции"
]

# 2. Данные для второй таблицы "ОтметкаФакта.csv"
fakt_file = "ОтметкаФакта.csv" # <--- Убедитесь, что это правильное имя файла
fakt_column_names = [
    "Дата закрытия операции",
    "Номер заказа на производство",
    "Артикул конечного изделия",
    "Наименование конечного изделия",
    "Наименование станции",
    "Артикул номенклатуры",
    "Наименование номенклатуры",
    "Закрытая трудоемкость",
    "Подразделение выполнения операции",
    "ШГО – вид оборудования",
    "Рабочий станок",
    "Номер операции",
    "Название операции"
]


# Обработка первой таблицы
df_srez = process_dataframe(srez_file, srez_column_names)
df_srez.to_csv("cleaned_srez.csv", index=False, encoding="utf-8-sig")
print(f"\nОбработанные данные из '{srez_file}' сохранены в 'cleaned_srez.csv'")


# Обработка второй таблицы
df_fakt = process_dataframe(fakt_file, fakt_column_names)
df_fakt.to_csv("cleaned_fakt.csv", index=False, encoding="utf-8-sig")
print(f"\nОбработанные данные из '{fakt_file}' сохранены в 'cleaned_fakt.csv'")

print("\n--- Все таблицы обработаны и сохранены. ---")