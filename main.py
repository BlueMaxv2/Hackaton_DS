import datetime
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import logging

logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)


def process_dataframe(file_path, column_names):
    print(f"\n--- Обработка файла: {file_path} ---")

    try:
        df = pd.read_csv(
            file_path,
            sep="^",
            encoding="cp1251",
            on_bad_lines="skip",
            header=None,
            decimal=','
        )
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден. Убедитесь, что он находится в той же директории, что и скрипт.")
        return pd.DataFrame()

    df.columns = column_names
    print(f"Исходное количество строк: {len(df)}")

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    initial_total_nan = df.isnull().sum().sum()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace('', np.nan)
    final_total_nan = df.isnull().sum().sum()

    df.drop_duplicates(inplace=True)

    rows_before_na_drop = len(df)
    df.dropna(inplace=True)
    print(f"Итоговое количество строк: {len(df)}")

    df.columns = df.columns.str.strip()

    date_cols = [
        "Дата записи среза",
        "Дата и время начала операции",
        "Дата и время окончания операции",
        "Дата закрытия операции"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d.%m.%Y %H:%M:%S", errors="coerce")
            df.dropna(subset=[col], inplace=True)

    numeric_cols_float = [
        "Общая трудоемкость",
        "Остаточная трудоемкость",
        "Закрытая трудоемкость"
    ]

    numeric_cols_int = [
        "Количество деталей",
        "Количество рабочих на операцию"
    ]

    for col in numeric_cols_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            df.dropna(subset=[col], inplace=True)

    for col in numeric_cols_int:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype('Int64')
            df.dropna(subset=[col], inplace=True)

    print(f"\nИнформация о данных после предобработки для {file_path}:")
    print(df.info())

    print(f"\nПервые строки таблицы после предобработки для {file_path}:")
    print(df.head())
    return df


def extract_department_info(department_string):
    if not isinstance(department_string, str):
        return None, None

    cleaned_string = re.sub(r'\s*\(.*\)', '', department_string).strip()

    main_department = None
    sub_department = None

    match_участок = re.search(r'(участок\s*(?:№)?\s*\d+\b)', cleaned_string, re.IGNORECASE)

    if match_участок:
        sub_department = match_участок.group(0).replace('№', '').strip()

        main_part = cleaned_string.replace(match_участок.group(0), '').strip()

        if ',' in main_part:
            main_department = main_part.split(',')[0].strip()
        else:
            main_department = main_part
    else:
        if ',' in cleaned_string:
            main_department = cleaned_string.split(',')[0].strip()
        else:
            main_department = cleaned_string

    if main_department:
        main_department = re.sub(r'\s+', ' ', main_department).strip()
    if sub_department:
        sub_department = re.sub(r'\s+', ' ', sub_department).strip()

    return main_department, sub_department


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

fakt_file = "ОтметкаФакта.csv"
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

df_srez = process_dataframe(srez_file, srez_column_names)
if not df_srez.empty:
    df_srez.to_csv("cleaned_srez.csv", index=False, encoding="utf-8-sig")
    print(f"\nОбработанные данные из '{srez_file}' сохранены в 'cleaned_srez.csv'")

df_fakt = process_dataframe(fakt_file, fakt_column_names)
if not df_fakt.empty:
    df_fakt.to_csv("cleaned_fakt.csv", index=False, encoding="utf-8-sig")
    print(f"\nОбработанные данные из '{fakt_file}' сохранены в 'cleaned_fakt.csv'")

print("\n--- Начало анализа количества операций по цехам/отделам ---")

if not df_srez.empty and 'Подразделение выполнения операции плановое' in df_srez.columns:
    print("\n--- Анализ операций из 'Срезы2024_2025.csv' ---")
    df_srez[['main_department', 'sub_department']] = df_srez['Подразделение выполнения операции плановое'].apply(
        lambda x: pd.Series(extract_department_info(x))
    )

    main_dept_counts_srez = df_srez['main_department'].value_counts()
    print("\nКоличество операций по основным цехам/отделам (Срезы):")
    for dept, count in main_dept_counts_srez.items():
        if pd.notna(dept):
            print(f"{dept} - {count} операций")

    sub_dept_counts_srez = df_srez.dropna(subset=['main_department', 'sub_department']).groupby(
        ['main_department', 'sub_department']).size().sort_values(ascending=False)
    print("\nКоличество операций по участкам (Срезы):")
    for (main_dept, sub_dept), count in sub_dept_counts_srez.items():
        if pd.notna(main_dept) and pd.notna(sub_dept):
            print(f"{main_dept}, {sub_dept} - {count} операций")
else:
    print(
        "DataFrame 'df_srez' пуст или не содержит колонки 'Подразделение выполнения операции плановое'. Анализ невозможен.")

if not df_fakt.empty and 'Подразделение выполнения операции' in df_fakt.columns:
    print("\n--- Анализ операций из 'ОтметкаФакта.csv' ---")
    df_fakt[['main_department', 'sub_department']] = df_fakt['Подразделение выполнения операции'].apply(
        lambda x: pd.Series(extract_department_info(x))
    )

    main_dept_counts_fakt = df_fakt['main_department'].value_counts()
    print("\nКоличество операций по основным цехам/отделам (ОтметкаФакта):")
    for dept, count in main_dept_counts_fakt.items():
        if pd.notna(dept):
            print(f"{dept} - {count} операций")

    sub_dept_counts_fakt = df_fakt.dropna(subset=['main_department', 'sub_department']).groupby(
        ['main_department', 'sub_department']).size().sort_values(ascending=False)
    print("\nКоличество операций по участкам (ОтметкаФакта):")
    for (main_dept, sub_dept), count in sub_dept_counts_fakt.items():
        if pd.notna(main_dept) and pd.notna(sub_dept):
            print(f"{main_dept}, {sub_dept} - {count} операций")
else:
    print("DataFrame 'df_fakt' пуст или не содержит колонки 'Подразделение выполнения операции'. Анализ невозможен.")

print("\n--- Анализ завершен. ---")

print("\n--- Анализ наиболее часто производимых изделий и деталей ---")

if not df_srez.empty:
    print("\n--- Анализ 'Срезы2024_2025.csv': Изделия и детали ---")

    if 'Артикул конечного изделия' in df_srez.columns:
        product_counts_srez = df_srez['Артикул конечного изделия'].value_counts()
        print("\n10 наиболее часто производимых конечных изделий (Срезы):")
        for product, count in product_counts_srez.head(10).items():
            if pd.notna(product):
                print(f"{product} - {count} раз")
    else:
        print("Колонка 'Артикул конечного изделия' не найдена в df_srez.")

    if 'Артикул номенклатуры' in df_srez.columns:
        nomenclature_counts_srez = df_srez['Артикул номенклатуры'].value_counts()
        print("\n10 наиболее часто производимых номенклатур (деталей) (Срезы):")
        for item, count in nomenclature_counts_srez.head(10).items():
            if pd.notna(item):
                print(f"{item} - {count} раз")
    else:
        print("Колонка 'Артикул номенклатуры' не найдена в df_srez.")
else:
    print("DataFrame 'df_srez' пуст. Анализ изделий и деталей невозможен.")

if not df_fakt.empty:
    print("\n--- Анализ 'ОтметкаФакта.csv': Изделия и детали ---")

    if 'Артикул конечного изделия' in df_fakt.columns:
        product_counts_fakt = df_fakt['Артикул конечного изделия'].value_counts()
        print("\n10 наиболее часто производимых конечных изделий (ОтметкаФакта):")
        for product, count in product_counts_fakt.head(10).items():
            if pd.notna(product):
                print(f"{product} - {count} раз")
    else:
        print("Колонка 'Артикул конечного изделия' не найдена в df_fakt.")

    if 'Артикул номенклатуры' in df_fakt.columns:
        nomenclature_counts_fakt = df_fakt['Артикул номенклатуры'].value_counts()
        print("\nТоп-10 наиболее часто производимых номенклатур (деталей) (ОтметкаФакта):")
        for item, count in nomenclature_counts_fakt.head(10).items():
            if pd.notna(item):
                print(f"{item} - {count} раз")
    else:
        print("Колонка 'Артикул номенклатуры' не найдена в df_fakt.")
else:
    print("DataFrame 'df_fakt' пуст. Анализ изделий и деталей невозможен.")

print("\n--- Анализ наиболее часто производимых изделий и деталей завершен. ---")

print("\n--- Анализ загрузки ШГО – вид оборудования и его динамики ---")

if not df_srez.empty and 'ШГО – вид оборудования' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
    print("\n--- Анализ на основе 'Срезы2024_2025.csv' ---")

    df_srez_filtered_load = df_srez.dropna(subset=['ШГО – вид оборудования', 'Общая трудоемкость']).copy()

    if not df_srez_filtered_load.empty:
        load_by_equipment = df_srez_filtered_load.groupby('ШГО – вид оборудования')[
            'Общая трудоемкость'].sum().sort_values(ascending=False)

        if not load_by_equipment.empty:
            most_loaded_equipment = load_by_equipment.index[0]
            max_load = load_by_equipment.iloc[0]
            print(
                f"\nСамое загруженное 'ШГО – вид оборудования': '{most_loaded_equipment}' с общей трудоемкостью: {max_load:.2f}")

            dynamics_df = df_srez_filtered_load[
                df_srez_filtered_load['ШГО – вид оборудования'] == most_loaded_equipment
                ].copy()

            load_dynamics = dynamics_df.groupby(dynamics_df['Дата записи среза'].dt.date)[
                'Общая трудоемкость'].sum().sort_index()

            print(f"\nДинамика загрузки '{most_loaded_equipment}' по дате записи среза:")
            if not load_dynamics.empty:
                for date, total_load in load_dynamics.items():
                    print(f"Дата: {date.strftime('%d.%m.%Y')} - Трудоемкость: {total_load:.2f}")
            else:
                print(f"Нет данных о трудоемкости для '{most_loaded_equipment}' после фильтрации по дате.")
        else:
            print("Не удалось определить самое загруженное ШГО (возможно, нет данных после фильтрации).")
    else:
        print("DataFrame для анализа загрузки ШГО пуст после обработки NaN.")
else:
    print(
        "DataFrame 'df_srez' пуст или не содержит необходимые колонки ('ШГО – вид оборудования' или 'Общая трудоемкость') для анализа загрузки оборудования.")

print("\n--- Анализ загрузки ШГО завершен. ---")

print("\n--- Распределение типов операций ---")

if not df_srez.empty and 'Название операции' in df_srez.columns:
    print("\n--- Анализ 'Срезы2024_2025.csv': Распределение типов операций ---")
    operation_type_counts_srez = df_srez['Название операции'].value_counts()
    print("\nТоп-10 наиболее часто встречающихся типов операций (Срезы):")
    for op_type, count in operation_type_counts_srez.head(10).items():
        if pd.notna(op_type):
            print(f"{op_type} - {count} раз")
else:
    print(
        "DataFrame 'df_srez' пуст или колонка 'Название операции' не найдена. Распределение типов операций невозможно.")

if not df_fakt.empty and 'Название операции' in df_fakt.columns:
    print("\n--- Анализ 'ОтметкаФакта.csv': Распределение типов операций ---")
    operation_type_counts_fakt = df_fakt['Название операции'].value_counts()
    print("\n10 наиболее часто встречающихся типов операций (ОтметкаФакта):")
    for op_type, count in operation_type_counts_fakt.head(10).items():
        if pd.notna(op_type):
            print(f"{op_type} - {count} раз")
else:
    print(
        "DataFrame 'df_fakt' пуст или колонка 'Название операции' не найдена. Распределение типов операций невозможно.")

print("\n--- Распределение типов операций завершено. ---")

print("\n--- 10 типов операций по загруженности ---")

if not df_srez.empty and 'Название операции' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
    print("\n--- Анализ 'Срезы2024_2025.csv': Топ-10 типов операций по загруженности ---")
    top_operations_srez = df_srez.groupby('Название операции')['Общая трудоемкость'].sum().sort_values(ascending=False)

    print("\n10 типов операций по общей трудоемкости (Срезы):")
    for op_type, workload in top_operations_srez.head(10).items():
        if pd.notna(op_type):
            print(f"{op_type} - {workload:.2f} трудоемкости")
else:
    print(
        "DataFrame 'df_srez' пуст или отсутствуют колонки 'Название операции' или 'Общая трудоемкость'. Топ-10 типов операций по загруженности невозможно.")

if not df_fakt.empty and 'Название операции' in df_fakt.columns and 'Закрытая трудоемкость' in df_fakt.columns:
    print("\n--- Анализ 'ОтметкаФакта.csv': 10 типов операций по загруженности ---")
    top_operations_fakt = df_fakt.groupby('Название операции')['Закрытая трудоемкость'].sum().sort_values(
        ascending=False)

    print("\n10 типов операций по закрытой трудоемкости (ОтметкаФакта):")
    for op_type, workload in top_operations_fakt.head(10).items():
        if pd.notna(op_type):
            print(f"{op_type} - {workload:.2f} трудоемкости")
else:
    print(
        "DataFrame 'df_fakt' пуст или отсутствуют колонки 'Название операции' или 'Закрытая трудоемкость'. Топ-10 типов операций по загруженности невозможно.")

print("\n--- Анализ 10 типов операций по загруженности завершен. ---")

print("\n--- Обнаружение сезонности/циклов по датам ---")

if not df_srez.empty and 'Дата записи среза' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
    print("\n--- Анализ 'Срезы2024_2025.csv': Сезонность по общей трудоемкости ---")

    df_srez_time_series = df_srez.set_index('Дата записи среза').copy()

    daily_srez_workload = df_srez_time_series['Общая трудоемкость'].resample('D').sum()
    print("\nЕжедневная динамика общей трудоемкости (Срезы) - первые 10 дней:")
    print(daily_srez_workload.head(10))
    print("\nЕжедневная динамика общей трудоемкости (Срезы) - последние 10 дней:")
    print(daily_srez_workload.tail(10))

    weekly_srez_workload = df_srez_time_series['Общая трудоемкость'].resample('W').sum()
    print("\nЕженедельная динамика общей трудоемкости (Срезы) - первые 10 недель:")
    print(weekly_srez_workload.head(10))
    print("\nЕженедельная динамика общей трудоемкости (Срезы) - последние 10 недель:")
    print(weekly_srez_workload.tail(10))

    monthly_srez_workload = df_srez_time_series['Общая трудоемкость'].resample('ME').sum()
    print("\nЕжемесячная динамика общей трудоемкости (Срезы):")
    print(monthly_srez_workload)
else:
    print(
        "DataFrame 'df_srez' пуст или отсутствуют колонки 'Дата записи среза' или 'Общая трудоемкость'. Анализ сезонности невозможен.")

if not df_fakt.empty and 'Дата закрытия операции' in df_fakt.columns and 'Закрытая трудоемкость' in df_fakt.columns:
    print("\n--- Анализ 'ОтметкаФакта.csv': Сезонность по закрытой трудоемкости ---")

    df_fakt_time_series = df_fakt.set_index('Дата закрытия операции').copy()

    daily_fakt_workload = df_fakt_time_series['Закрытая трудоемкость'].resample('D').sum()
    print("\nЕжедневная динамика закрытой трудоемкости (ОтметкаФакта) - первые 10 дней:")
    print(daily_fakt_workload.head(10))
    print("\nЕжедневная динамика закрытой трудоемкости (ОтметкаФакта) - последние 10 дней:")
    print(daily_fakt_workload.tail(10))

    weekly_fakt_workload = df_fakt_time_series['Закрытая трудоемкость'].resample('W').sum()
    print("\nЕженедельная динамика закрытой трудоемкости (ОтметкаФакта) - первые 10 недель:")
    print(weekly_fakt_workload.head(10))
    print("\nЕженедельная динамика закрытой трудоемкости (ОтметкаФакта) - последние 10 недель:")
    print(weekly_fakt_workload.tail(10))

    monthly_fakt_workload = df_fakt_time_series['Закрытая трудоемкость'].resample('ME').sum()
    print("\nЕжемесячная динамика закрытой трудоемкости (ОтметкаФакта):")
    print(monthly_fakt_workload)
else:
    print(
        "DataFrame 'df_fakt' пуст или отсутствуют колонки 'Дата закрытия операции' или 'Закрытая трудоемкость'. Анализ сезонности невозможен.")

print("\n--- Анализ сезонности/циклов завершен. ---")

print("\n--- Расчет средней и медианной загрузки по участкам и ШГО ---")

if not df_srez.empty:
    print("\n--- Анализ 'Срезы2024_2025.csv' (Плановая загрузка) ---")

    if 'main_department' in df_srez.columns and 'sub_department' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
        print("\nСредняя и медианная загрузка по участкам (Срезы):")
        avg_median_workload_srez_dept = \
            df_srez.dropna(subset=['main_department', 'sub_department', 'Общая трудоемкость']).groupby(
                ['main_department', 'sub_department'])['Общая трудоемкость'].agg(['mean', 'median']).sort_values(
                by='mean',
                ascending=False)
        print(avg_median_workload_srez_dept.to_string())
    else:
        print(
            "Недостаточно данных для расчета средней и медианной загрузки по участкам в df_srez. Убедитесь, что колонки 'main_department', 'sub_department' и 'Общая трудоемкость' существуют и не пусты.")

    if 'ШГО – вид оборудования' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
        print("\nСредняя и медианная загрузка по ШГО (Срезы):")
        avg_median_workload_srez_shgo = \
            df_srez.dropna(subset=['ШГО – вид оборудования', 'Общая трудоемкость']).groupby('ШГО – вид оборудования')[
                'Общая трудоемкость'].agg(['mean', 'median']).sort_values(by='mean', ascending=False)
        print(avg_median_workload_srez_shgo.to_string())
    else:
        print(
            "Недостаточно данных для расчета средней и медианной загрузки по ШГО в df_srez. Убедитесь, что колонки 'ШГО – вид оборудования' и 'Общая трудоемкость' существуют и не пусты.")
else:
    print("DataFrame 'df_srez' пуст. Расчет средней и медианной загрузки невозможен.")

if not df_fakt.empty:
    print("\n--- Анализ 'ОтметкаФакта.csv' (Фактическая загрузка) ---")

    if 'main_department' in df_fakt.columns and 'sub_department' in df_fakt.columns and 'Закрытая трудоемкость' in df_fakt.columns:
        print("\nСредняя и медианная загрузка по участкам (ОтметкаФакта):")
        avg_median_workload_fakt_dept = \
            df_fakt.dropna(subset=['main_department', 'sub_department', 'Закрытая трудоемкость']).groupby(
                ['main_department', 'sub_department'])['Закрытая трудоемкость'].agg(['mean', 'median']).sort_values(
                by='mean', ascending=False)
        print(avg_median_workload_fakt_dept.to_string())
    else:
        print(
            "Недостаточно данных для расчета средней и медианной загрузки по участкам в df_fakt. Убедитесь, что колонки 'main_department', 'sub_department' и 'Закрытая трудоемкость' существуют и не пусты.")

    if 'ШГО – вид оборудования' in df_fakt.columns and 'Закрытая трудоемкость' in df_fakt.columns:
        print("\nСредняя и медианная загрузка по ШГО (ОтметкаФакта):")
        avg_median_workload_fakt_shgo = \
            df_fakt.dropna(subset=['ШГО – вид оборудования', 'Закрытая трудоемкость']).groupby(
                'ШГО – вид оборудования')[
                'Закрытая трудоемкость'].agg(['mean', 'median']).sort_values(by='mean', ascending=False)
        print(avg_median_workload_fakt_shgo.to_string())
    else:
        print(
            "Недостаточно данных для расчета средней и медианной загрузки по ШГО в df_fakt. Убедитесь, что колонки 'ШГО – вид оборудования' и 'Закрытая трудоемкость' существуют и не пусты.")
else:
    print("DataFrame 'df_fakt' пуст. Расчет средней и медианной загрузки невозможен.")

print("\n--- Расчет средней и медианной загрузки завершен. ---")

print("\n--- Анализ тенденции сдвига даты начала операции по срезам плана ---")

if not df_srez.empty and \
        'Номер заказа на производство' in df_srez.columns and \
        'Номер операции' in df_srez.columns and \
        'Дата записи среза' in df_srez.columns and \
        'Дата и время начала операции' in df_srez.columns and \
        'Название операции' in df_srez.columns and \
        'Артикул номенклатуры' in df_srez.columns:

    print("Начало анализа сдвигов дат начала операций")

    df_srez_sorted = df_srez.sort_values(by=[
        'Номер заказа на производство',
        'Номер операции',
        'Артикул номенклатуры',
        'Дата записи среза'
    ]).copy()

    df_srez_sorted['operation_id'] = df_srez_sorted['Номер заказа на производство'].astype(str) + '_' + \
                                     df_srez_sorted['Номер операции'].astype(str) + '_' + \
                                     df_srez_sorted['Артикул номенклатуры'].astype(str)

    df_srez_sorted['prev_slice_date'] = df_srez_sorted.groupby('operation_id')['Дата записи среза'].shift(1)
    df_srez_sorted['prev_start_date'] = df_srez_sorted.groupby('operation_id')['Дата и время начала операции'].shift(1)

    shifts_df = df_srez_sorted.dropna(
        subset=['prev_slice_date', 'prev_start_date']).copy()  # Переименовал shifts_df.rename на после dropna

    shifts_df.rename(columns={
        'prev_slice_date': 'Предыдущая дата среза',
        'prev_start_date': 'Предыдущая дата начала операции'
    }, inplace=True)

    if not shifts_df.empty:
        shifts_df['Сдвиг в днях'] = (
                shifts_df['Дата и время начала операции'] - shifts_df['Предыдущая дата начала операции']).dt.days

        shifts_df['Направление сдвига'] = shifts_df['Сдвиг в днях'].apply(
            lambda x: 'Опережение' if x < 0 else ('Задержка' if x > 0 else 'Без сдвига')
        )

        operations_with_shifts = shifts_df[shifts_df['Сдвиг в днях'] != 0].copy()

        if not operations_with_shifts.empty:
            top_20_shifted_operations = operations_with_shifts.sort_values(
                by='Сдвиг в днях',
                key=abs,
                ascending=False
            ).head(20)

            print("\n20 операций с наибольшими сдвигами даты начала (отклоняющиеся от нормы):")
            print(top_20_shifted_operations[[
                'Номер заказа на производство',
                'Номер операции',
                'Название операции',
                'Артикул номенклатуры',
                'Дата записи среза',
                'Предыдущая дата среза',
                'Дата и время начала операции',
                'Предыдущая дата начала операции',
                'Сдвиг в днях',
                'Направление сдвига'
            ]].to_string(index=False))
        else:
            print("Не обнаружено операций с ненулевым сдвигом даты начала.")

        avg_shift = shifts_df['Сдвиг в днях'].mean()
        median_shift = shifts_df['Сдвиг в днях'].median()
        print(f"\nСредний сдвиг даты начала операции (по всем проанализированным срезам): {avg_shift:.2f} дней")
        print(f"Медианный сдвиг даты начала операции (по всем проанализированным срезам): {median_shift:.2f} дней")

        shift_distribution = shifts_df['Направление сдвига'].value_counts(normalize=True) * 100
        print("\nРаспределение сдвигов:")
        print(shift_distribution.to_string())

        # Этот блок уже был корректным, он использует nlargest для получения самых больших сдвигов
        top_shifted_ops = shifts_df.loc[shifts_df['Сдвиг в днях'].abs().nlargest(10).index]
        print("\n10 операций с наибольшим абсолютным сдвигом:")
        print(top_shifted_ops[[
            'Номер заказа на производство',
            'Номер операции',
            'Название операции',
            'Артикул номенклатуры',
            'Дата записи среза',
            'Предыдущая дата среза',
            'Дата и время начала операции',
            'Предыдущая дата начала операции',
            'Сдвиг в днях',
            'Направление сдвига'
        ]].to_string())

    else:
        print(
            "Недостаточно данных в df_srez для анализа сдвигов дат начала операций после фильтрации. Убедитесь, что для каждой операции есть несколько срезов.")
else:
    print(
        "DataFrame 'df_srez' пуст или отсутствуют необходимые колонки для анализа сдвигов (например, 'Номер заказа на производство', 'Номер операции', 'Дата записи среза', 'Дата и время начала операции', 'Название операции', 'Артикул номенклатуры').")

print("\n--- Анализ сдвигов и задержек завершен. ---")

print("\n--- Поиск необычных комбинаций 'Название операции' и 'ШГО – вид оборудования' ---")

if not df_srez.empty and \
        'Название операции' in df_srez.columns and \
        'ШГО – вид оборудования' in df_srez.columns and \
        'Рабочий станок' in df_srez.columns:

    df_anomaly_analysis = df_srez.dropna(subset=[
        'Название операции',
        'ШГО – вид оборудования',
        'Рабочий станок'
    ]).copy()

    if not df_anomaly_analysis.empty:
        operation_shgo_machine_counts = df_anomaly_analysis.groupby([
            'Название операции',
            'ШГО – вид оборудования',
            'Рабочий станок'
        ]).size().reset_index(name='Количество операций').sort_values(by='Количество операций', ascending=True)

        print("\n20 самых редких комбинаций 'Название операции', 'ШГО – вид оборудования' и 'Рабочий станок':")
        print(operation_shgo_machine_counts.head(20).to_string(index=False))

        print(
            "\nЭти комбинации встречаются редко и могут быть потенциальными аномалиями или ошибками, которые требуют ручной проверки.")
    else:
        print(
            "Нет достаточных данных для анализа комбинаций операции, ШГО и рабочего станка после фильтрации строк с отсутствующими значениями.")
else:
    print(
        "DataFrame 'df_srez' пуст или отсутствуют колонки 'Название операции', 'ШГО – вид оборудования' или 'Рабочий станок'. Поиск необычных комбинаций невозможен.")

print("\n--- Поиск необычных комбинаций завершен. ---")

print("\n--- Анализ загрузки ШГО по периодам для поиска оптимального времени ---")

if not df_srez.empty and \
        'Дата записи среза' in df_srez.columns and \
        'ШГО – вид оборудования' in df_srez.columns and \
        'Общая трудоемкость' in df_srez.columns:

    df_shgo_period_analysis = df_srez.dropna(subset=[
        'Дата записи среза',
        'ШГО – вид оборудования',
        'Общая трудоемкость'
    ]).copy()

    if not df_shgo_period_analysis.empty:
        df_shgo_period_analysis['Месяц'] = df_shgo_period_analysis['Дата записи среза'].dt.to_period('M')

        print("\nРасчет средней ежемесячной загрузки по 'ШГО – вид оборудования':")

        monthly_shgo_load = df_shgo_period_analysis.groupby([
            'ШГО – вид оборудования',
            'Месяц'
        ])['Общая трудоемкость'].sum().unstack(fill_value=0)

        monthly_shgo_load['Общая Средняя Загрузка'] = monthly_shgo_load.mean(axis=1)
        monthly_shgo_load = monthly_shgo_load.sort_values(by='Общая Средняя Загрузка', ascending=False)

        print(monthly_shgo_load.head(20).to_string())
    else:
        print(
            "Нет достаточных данных для анализа загрузки ШГО по периодам после фильтрации строк с отсутствующими значениями.")
else:
    print(
        "DataFrame 'df_srez' пуст или отсутствуют необходимые колонки ('Дата записи среза', 'ШГО – вид оборудования', 'Общая трудоемкость'). Анализ загрузки по периодам невозможен.")

print("\n--- Анализ загрузки ШГО по периодам завершен. ---")

print("\n--- Анализ закрытия факта для поиска тенденции развития организации ---")

if not df_fakt.empty and \
        'Дата закрытия операции' in df_fakt.columns and \
        'Закрытая трудоемкость' in df_fakt.columns:

    df_fakt_trend_analysis = df_fakt.dropna(subset=[
        'Дата закрытия операции',
        'Закрытая трудоемкость'
    ]).copy()

    if not df_fakt_trend_analysis.empty:
        df_fakt_trend_analysis = df_fakt_trend_analysis.set_index('Дата закрытия операции')

        print("\nЕженедельная динамика закрытой трудоемкости:")
        weekly_fakt_workload_trend = df_fakt_trend_analysis['Закрытая трудоемкость'].resample('W').sum()
        print(weekly_fakt_workload_trend.to_string())

        print("\nЕжемесячная динамика закрытой трудоемкости:")
        monthly_fakt_workload_trend = df_fakt_trend_analysis['Закрытая трудоемкость'].resample('ME').sum()
        print(monthly_fakt_workload_trend.to_string())

    else:
        print(
            "Нет достаточных данных для анализа тенденций закрытия факта после фильтрации строк с отсутствующими значениями.")
else:
    print(
        "DataFrame 'df_fakt' пуст или отсутствуют необходимые колонки ('Дата закрытия операции', 'Закрытая трудоемкость'). Анализ тенденций закрытия факта невозможен.")

print("\n--- Анализ тенденций закрытия факта завершен. ---")

print("\n--- Прогноз выполнимости среза от 01.02.2025 на основе временных рядов ---")

try:
    from prophet import Prophet

    prophet_available = True
except ImportError:
    prophet_available = False

try:
    import matplotlib.pyplot as plt

    matplotlib_available = True
except ImportError:
    matplotlib_available = False

if prophet_available and not df_srez.empty and \
        'Дата записи среза' in df_srez.columns and \
        'Общая трудоемкость' in df_srez.columns:

    df_forecast_analysis = df_srez.dropna(subset=[
        'Дата записи среза',
        'Общая трудоемкость'
    ]).copy()

    df_forecast_analysis['Дата записи среза'] = pd.to_datetime(df_forecast_analysis['Дата записи среза'],
                                                               errors='coerce')
    df_forecast_analysis.dropna(subset=['Дата записи среза'], inplace=True)  # ИСПРАВЛЕНИЕ: БЫЛО 'Дата записи среza'
    df_forecast_analysis = df_forecast_analysis[df_forecast_analysis['Общая трудоемкость'] >= 0]

    if not df_forecast_analysis.empty:
        monthly_workload = df_forecast_analysis.groupby(
            df_forecast_analysis['Дата записи среза'].dt.to_period('M')).sum(
            numeric_only=True
        ).reset_index()
        monthly_workload['Дата записи среза'] = monthly_workload['Дата записи среза'].dt.to_timestamp()
        monthly_workload = monthly_workload[['Дата записи среза', 'Общая трудоемкость']]
        monthly_workload.columns = ['ds', 'y']

        print("\nСтатистика агрегированной ЕЖЕМЕСЯЧНОЙ трудоемкости (monthly_workload для Prophet):")
        print(monthly_workload['y'].describe().to_string())
        print(f"Первые 5 значений monthly_workload:\n{monthly_workload.head().to_string(index=False)}")
        print(f"Последние 5 значений monthly_workload:\n{monthly_workload.tail().to_string(index=False)}")

        if matplotlib_available:
            print("\nВизуализация исторических данных 'Общая трудоемкость' (ЕЖЕМЕСЯЧНО):")
            plt.figure(figsize=(12, 6))
            plt.plot(monthly_workload['ds'], monthly_workload['y'])
            plt.title('Историческая Общая Трудоемкость (ЕЖЕМЕСЯЧНО) по датам')
            plt.xlabel('Дата')
            plt.ylabel('Общая трудоемкость')
            plt.grid(True)
            plt.show()

        train_data = monthly_workload.copy()

        if not train_data.empty:
            print(f"\nПоследняя дата в обучающих данных: {train_data['ds'].max().strftime('%Y-%m-%d')}")

        if train_data.empty:
            print("Нет данных для обучения модели. Прогноз невозможен.")
        else:
            model = Prophet(
                weekly_seasonality=False,  # Отключаем, если прогнозируем месячные данные
                daily_seasonality=False,
                yearly_seasonality=False,  # <--- ИЗМЕНЕНИЕ ЗДЕСЬ: ОТКЛЮЧАЕМ ГОДОВУЮ СЕЗОННОСТЬ
                changepoint_prior_scale=0.05
            )
            model.fit(train_data)

            # Прогнозируем на 3 месяца вперед с месячной частотой ('MS' - Month Start)
            future_dates = model.make_future_dataframe(periods=3, include_history=False, freq='MS')

            if future_dates.empty or future_dates['ds'].min() > pd.to_datetime('2025-05-01'):
                print(
                    f"Не удалось создать осмысленные будущие даты для прогноза. Проверьте диапазон исторических данных. Последняя дата в обучающей выборке: {train_data['ds'].max() if not train_data.empty else 'N/A'}")
            else:
                forecast = model.predict(future_dates)

                forecast['yhat'] = np.maximum(0, forecast['yhat'])
                forecast['yhat_lower'] = np.maximum(0, forecast['yhat_lower'])
                forecast['yhat_upper'] = np.maximum(0, forecast['yhat_upper'])

                forecast_start_date_target = pd.to_datetime('2025-02-01')
                if forecast['ds'].min() > forecast_start_date_target:
                    print(
                        f"ВНИМАНИЕ: Прогноз начинается с {forecast['ds'].min().strftime('%Y-%m-%d')}, а не с {forecast_start_date_target.strftime('%Y-%m-%d')}, так как обучающие данные заканчиваются раньше.")
                elif forecast['ds'].min() < forecast_start_date_target:
                    forecast = forecast[forecast['ds'] >= forecast_start_date_target]

                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_display.columns = ['Дата', 'Прогноз трудоемкости', 'Нижняя граница (95% ДИ)',
                                            'Верхняя граница (95% ДИ)']

                print(
                    f"\nПрогноз ОБЩЕЙ ТРУДОЕМКОСТИ на ближайшие {len(forecast_display)} МЕСЯЦЕВ (начиная с первой доступной даты после 01.02.2025):")
                print(forecast_display.to_string(index=False))

    else:
        print("Нет достаточных данных для построения прогноза после фильтрации строк с отсутствующими значениями.")
else:
    if not prophet_available:
        print("Прогноз не выполнен, так как библиотека 'prophet' не установлена.")
    else:
        print(
            "DataFrame 'df_srez' пуст или отсутствуют необходимые колонки ('Дата записи среза', 'Общая трудоемкость'). Прогноз невозможен.")

print("\n--- Прогноз выполнимости среза завершен. ---")
