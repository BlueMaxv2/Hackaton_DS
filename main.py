import datetime
import pandas as pd
import numpy as np
import re # Импортируем модуль для регулярных выражений

def process_dataframe(file_path, column_names):
    """
    Загружает, очищает и преобразует типы данных в DataFrame.

    Args:
        file_path (str): Путь к CSV-файлу.
        column_names (list): Список корректных названий колонок.

    Returns:
        pandas.DataFrame: Обработанный DataFrame.
    """
    print(f"\n--- Обработка файла: {file_path} ---")

    try:
        # Загрузка данных
        df = pd.read_csv(
            file_path,
            sep="^",
            encoding="cp1251",
            on_bad_lines="skip", # Пропускать строки с ошибками парсинга
            header=None,         # Указываем, что заголовков нет
            decimal=','          # Указываем, что десятичный разделитель - запятая
        )
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден. Убедитесь, что он находится в той же директории, что и скрипт.")
        return pd.DataFrame() # Возвращаем пустой DataFrame в случае ошибки

    # Присваиваем новые названия колонкам
    df.columns = column_names
    print(f"Исходное количество строк: {len(df)}")

    # Очистка лишних пробелов в строковых данных
    print("Начало обрезки лишних пробелов в строковых колонках.")
    for col in df.select_dtypes(include=['object']).columns:
        # Применяем strip только к строкам, игнорируя NaN и другие типы
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    print("Произведена обрезка лишних пробелов в строковых колонках.")

    # Преобразование пустых строк в NaN
    print("Начало замены пустых строк на NaN.")
    initial_total_nan = df.isnull().sum().sum() # Общее количество NaN до замены пустых строк
    for col in df.select_dtypes(include=['object']).columns:
        # ИСПРАВЛЕНИЕ: присваиваем результат обратно в колонку, вместо inplace=True
        # Это решает FutureWaring и гарантирует изменение DataFrame
        df[col] = df[col].replace('', np.nan)
    final_total_nan = df.isnull().sum().sum() # Общее количество NaN после замены пустых строк
    print(f"Пустые строки в текстовых колонках заменены на NaN. Общее количество NaN увеличилось на: {final_total_nan - initial_total_nan}")

    # Удаление дубликатов
    df.drop_duplicates(inplace=True)
    print(f"Количество строк после удаления дубликатов: {len(df)}")

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
            df.dropna(subset=[col], inplace=True) # Удаляем строки, где дата не смогла преобразоваться
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
            df.dropna(subset=[col], inplace=True) # Удаляем строки, где число не смогло преобразоваться
        # else:
            # print(f"Предупреждение: Колонка '{col}' (ожидался float) не найдена.")

    for col in numeric_cols_int:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype('Int64') # Int64 поддерживает NaN
            # После преобразования в numeric, если остаются NaN, то удаляем
            df.dropna(subset=[col], inplace=True) # Удаляем строки, где число не смогло преобразоваться
        # else:
            # print(f"Предупреждение: Колонка '{col}' (ожидался Int64) не найдена.")

    print(f"\nИнформация о данных после предобработки для {file_path}:")
    print(df.info())

    print(f"\nПервые строки таблицы после предобработки для {file_path}:")
    print(df.head())

    return df

# --- Функция для извлечения информации о подразделениях ---
def extract_department_info(department_string):
    """
    Извлекает основное подразделение (цех/отдел) и участок из строки.
    """
    if not isinstance(department_string, str):
        return None, None

    # 1. Удалить текст в скобках (например, "(до 01.09.2023: ...)") и любые лишние пробелы вокруг них
    cleaned_string = re.sub(r'\s*\(.*\)', '', department_string).strip()

    main_department = None
    sub_department = None

    # 2. Ищем паттерн "участок" (участок, участок №, участок №N, участок N)
    # Используем \b для обозначения границы слова, чтобы не захватывать "участокник"
    match_участок = re.search(r'(участок\s*(?:№)?\s*\d+\b)', cleaned_string, re.IGNORECASE)

    if match_участок:
        # Извлекаем найденный участок и нормализуем его (убираем "№")
        sub_department = match_участок.group(0).replace('№', '').strip()

        # Удаляем найденный участок из строки, чтобы получить основное подразделение
        main_part = cleaned_string.replace(match_участок.group(0), '').strip()

        # Если в оставшейся части есть запятая, берем часть до нее как основное подразделение
        if ',' in main_part:
            main_department = main_part.split(',')[0].strip()
        else:
            main_department = main_part
    else:
        # Если нет паттерна "участок", то вся строка (или часть до первой запятой) - это основное подразделение
        if ',' in cleaned_string:
            main_department = cleaned_string.split(',')[0].strip()
        else:
            main_department = cleaned_string

    # 3. Нормализация: убираем множественные пробелы и ведущие/хвостовые пробелы
    if main_department:
        main_department = re.sub(r'\s+', ' ', main_department).strip()
    if sub_department:
        sub_department = re.sub(r'\s+', ' ', sub_department).strip()

    return main_department, sub_department

# --- Основная часть скрипта ---

# 1. Данные для первой таблицы "Срезы2024_2025.csv"
srez_file = "Срезы2024_2025.csv"
srez_column_names = [
    "Дата записи среза",
    "Номер заказа на производство",
    "Артикул конечного изделия", # Колонка для анализа
    "Наименование конечного изделия",
    "Наименование станции",
    "Артикул номенклатуры", # Колонка для анализа
    "Наименование номенклатуры",
    "Количество деталей",
    "Номер операции",
    "Название операции", # Колонка для анализа типов операций и загруженности
    "Общая трудоемкость", # Колонка для анализа загруженности
    "Остаточная трудоемкость",
    "Количество рабочих на операцию",
    "Подразделение выполнения операции плановое", # Колонка для анализа
    "ШГО – вид оборудования", # Колонка для анализа
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
    "Артикул конечного изделия", # Колонка для анализа
    "Наименование конечного изделия",
    "Наименование станции",
    "Артикул номенклатуры", # Колонка для анализа
    "Наименование номенклатуры",
    "Закрытая трудоемкость", # Колонка для анализа загруженности
    "Подразделение выполнения операции", # Колонка для анализа
    "ШГО – вид оборудования", # Колонка для анализа (если есть, используется для проверки)
    "Рабочий станок",
    "Номер операции",
    "Название операции" # Колонка для анализа типов операций и загруженности
]


# Обработка первой таблицы
df_srez = process_dataframe(srez_file, srez_column_names)
if not df_srez.empty: # Сохраняем только если DataFrame не пустой (т.е. файл найден и обработан)
    df_srez.to_csv("cleaned_srez.csv", index=False, encoding="utf-8-sig")
    print(f"\nОбработанные данные из '{srez_file}' сохранены в 'cleaned_srez.csv'")


# Обработка второй таблицы
df_fakt = process_dataframe(fakt_file, fakt_column_names)
if not df_fakt.empty: # Сохраняем только если DataFrame не пустой
    df_fakt.to_csv("cleaned_fakt.csv", index=False, encoding="utf-8-sig")
    print(f"\nОбработанные данные из '{fakt_file}' сохранены в 'cleaned_fakt.csv'")

print("\n--- Все таблицы обработаны и сохранены (если файлы были найдены). ---")

# --- Анализ количества операций по цехам/отделам ---

print("\n--- Начало анализа количества операций по цехам/отделам ---")

# Анализ df_srez
if not df_srez.empty and 'Подразделение выполнения операции плановое' in df_srez.columns:
    print("\n--- Анализ операций из 'Срезы2024_2025.csv' ---")
    # Применяем функцию извлечения информации и создаем новые колонки
    df_srez[['main_department', 'sub_department']] = df_srez['Подразделение выполнения операции плановое'].apply(
        lambda x: pd.Series(extract_department_info(x)) # Используем pd.Series для возврата нескольких значений
    )

    # Считаем операции для основных подразделений
    main_dept_counts_srez = df_srez['main_department'].value_counts()
    print("\nКоличество операций по основным цехам/отделам (Срезы):")
    for dept, count in main_dept_counts_srez.items():
        if pd.notna(dept): # Пропускаем None/NaN подразделения
            print(f"{dept} - {count} операций")

    # Считаем операции для подразделений с участками
    # Группируем по основному подразделению и участку, считаем размер группы
    sub_dept_counts_srez = df_srez.dropna(subset=['main_department', 'sub_department']).groupby(['main_department', 'sub_department']).size().sort_values(ascending=False)
    print("\nКоличество операций по участкам (Срезы):")
    for (main_dept, sub_dept), count in sub_dept_counts_srez.items():
        if pd.notna(main_dept) and pd.notna(sub_dept): # Пропускаем None/NaN
            print(f"{main_dept}, {sub_dept} - {count} операций")
else:
    print("DataFrame 'df_srez' пуст или не содержит колонки 'Подразделение выполнения операции плановое'. Анализ невозможен.")


# Анализ df_fakt
if not df_fakt.empty and 'Подразделение выполнения операции' in df_fakt.columns:
    print("\n--- Анализ операций из 'ОтметкаФакта.csv' ---")
    # Применяем функцию извлечения информации и создаем новые колонки
    df_fakt[['main_department', 'sub_department']] = df_fakt['Подразделение выполнения операции'].apply(
        lambda x: pd.Series(extract_department_info(x))
    )

    # Считаем операции для основных подразделений
    main_dept_counts_fakt = df_fakt['main_department'].value_counts()
    print("\nКоличество операций по основным цехам/отделам (ОтметкаФакта):")
    for dept, count in main_dept_counts_fakt.items():
        if pd.notna(dept):
            print(f"{dept} - {count} операций")

    # Считаем операции для подразделений с участками
    sub_dept_counts_fakt = df_fakt.dropna(subset=['main_department', 'sub_department']).groupby(['main_department', 'sub_department']).size().sort_values(ascending=False)
    print("\nКоличество операций по участкам (ОтметкаФакта):")
    for (main_dept, sub_dept), count in sub_dept_counts_fakt.items():
        if pd.notna(main_dept) and pd.notna(sub_dept):
            print(f"{main_dept}, {sub_dept} - {count} операций")
else:
    print("DataFrame 'df_fakt' пуст или не содержит колонки 'Подразделение выполнения операции'. Анализ невозможен.")

print("\n--- Анализ завершен. ---")

# --- Анализ наиболее часто производимых изделий и деталей ---

print("\n--- Анализ наиболее часто производимых изделий и деталей ---")

# Анализ df_srez
if not df_srez.empty:
    print("\n--- Анализ 'Срезы2024_2025.csv': Изделия и детали ---")

    # Анализ Артикул конечного изделия
    if 'Артикул конечного изделия' in df_srez.columns:
        product_counts_srez = df_srez['Артикул конечного изделия'].value_counts()
        print("\nТоп-10 наиболее часто производимых конечных изделий (Срезы):")
        for product, count in product_counts_srez.head(10).items():
            if pd.notna(product):
                print(f"{product} - {count} раз")
    else:
        print("Колонка 'Артикул конечного изделия' не найдена в df_srez.")

    # Анализ Артикул номенклатуры
    if 'Артикул номенклатуры' in df_srez.columns:
        nomenclature_counts_srez = df_srez['Артикул номенклатуры'].value_counts()
        print("\nТоп-10 наиболее часто производимых номенклатур (деталей) (Срезы):")
        for item, count in nomenclature_counts_srez.head(10).items():
            if pd.notna(item):
                print(f"{item} - {count} раз")
    else:
        print("Колонка 'Артикул номенклатуры' не найдена в df_srez.")
else:
    print("DataFrame 'df_srez' пуст. Анализ изделий и деталей невозможен.")


# Анализ df_fakt
if not df_fakt.empty:
    print("\n--- Анализ 'ОтметкаФакта.csv': Изделия и детали ---")

    # Анализ Артикул конечного изделия
    if 'Артикул конечного изделия' in df_fakt.columns:
        product_counts_fakt = df_fakt['Артикул конечного изделия'].value_counts()
        print("\nТоп-10 наиболее часто производимых конечных изделий (ОтметкаФакта):")
        for product, count in product_counts_fakt.head(10).items():
            if pd.notna(product):
                print(f"{product} - {count} раз")
    else:
        print("Колонка 'Артикул конечного изделия' не найдена в df_fakt.")

    # Анализ Артикул номенклатуры
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

# --- Анализ загрузки ШГО – вид оборудования и его динамики ---

print("\n--- Анализ загрузки ШГО – вид оборудования и его динамики ---")

if not df_srez.empty and 'ШГО – вид оборудования' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
    print("\n--- Анализ на основе 'Срезы2024_2025.csv' ---")

    # 1. Найти самое загруженное ШГО по общей трудоемкости
    # Убедимся, что 'Общая трудоемкость' является числовой колонкой и не содержит NaN перед суммированием
    df_srez_filtered_load = df_srez.dropna(subset=['ШГО – вид оборудования', 'Общая трудоемкость']).copy()

    if not df_srez_filtered_load.empty:
        # Считаем суммарную трудоемкость для каждого вида оборудования
        load_by_equipment = df_srez_filtered_load.groupby('ШГО – вид оборудования')['Общая трудоемкость'].sum().sort_values(ascending=False)

        if not load_by_equipment.empty:
            most_loaded_equipment = load_by_equipment.index[0]
            max_load = load_by_equipment.iloc[0]
            print(f"\nСамое загруженное 'ШГО – вид оборудования': '{most_loaded_equipment}' с общей трудоемкостью: {max_load:.2f}")

            # 2. Динамика загрузки этого ШГО по срезам
            # Фильтруем данные для самого загруженного оборудования
            dynamics_df = df_srez_filtered_load[
                df_srez_filtered_load['ШГО – вид оборудования'] == most_loaded_equipment
            ].copy()

            # Группируем по дате записи среза и суммируем трудоемкость
            # Используем .dt.date для группировки по дате без времени
            load_dynamics = dynamics_df.groupby(dynamics_df['Дата записи среза'].dt.date)['Общая трудоемкость'].sum().sort_index()

            print(f"\nДинамика загрузки '{most_loaded_equipment}' по дате записи среза:")
            if not load_dynamics.empty:
                for date, total_load in load_dynamics.items():
                    print(f"Дата: {date.strftime('%d.%m.%Y')} - Трудоемкость: {total_load:.2f}")
                print("\nДля визуализации этой динамики вы можете использовать библиотеку, такую как Matplotlib или Seaborn, построив линейный график с датой по оси X и трудоемкостью по оси Y.")
            else:
                print(f"Нет данных о трудоемкости для '{most_loaded_equipment}' после фильтрации по дате.")
        else:
            print("Не удалось определить самое загруженное ШГО (возможно, нет данных после фильтрации).")
    else:
        print("DataFrame для анализа загрузки ШГО пуст после обработки NaN.")
else:
    print("DataFrame 'df_srez' пуст или не содержит необходимые колонки ('ШГО – вид оборудования' или 'Общая трудоемкость') для анализа загрузки оборудования.")

print("\n--- Анализ загрузки ШГО завершен. ---")

# --- Распределение типов операций ---

print("\n--- Распределение типов операций ---")

if not df_srez.empty and 'Название операции' in df_srez.columns:
    print("\n--- Анализ 'Срезы2024_2025.csv': Распределение типов операций ---")
    operation_type_counts_srez = df_srez['Название операции'].value_counts()
    print("\nТоп-10 наиболее часто встречающихся типов операций (Срезы):")
    for op_type, count in operation_type_counts_srez.head(10).items():
        if pd.notna(op_type):
            print(f"{op_type} - {count} раз")
else:
    print("DataFrame 'df_srez' пуст или колонка 'Название операции' не найдена. Распределение типов операций невозможно.")

if not df_fakt.empty and 'Название операции' in df_fakt.columns:
    print("\n--- Анализ 'ОтметкаФакта.csv': Распределение типов операций ---")
    operation_type_counts_fakt = df_fakt['Название операции'].value_counts()
    print("\nТоп-10 наиболее часто встречающихся типов операций (ОтметкаФакта):")
    for op_type, count in operation_type_counts_fakt.head(10).items():
        if pd.notna(op_type):
            print(f"{op_type} - {count} раз")
else:
    print("DataFrame 'df_fakt' пуст или колонка 'Название операции' не найдена. Распределение типов операций невозможно.")

print("\n--- Распределение типов операций завершено. ---")

# --- Топ-10 типов операций по загруженности ---

print("\n--- Топ-10 типов операций по загруженности ---")

# Анализ df_srez
if not df_srez.empty and 'Название операции' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
    print("\n--- Анализ 'Срезы2024_2025.csv': Топ-10 типов операций по загруженности ---")
    # Группируем по названию операции и суммируем общую трудоемкость
    top_operations_srez = df_srez.groupby('Название операции')['Общая трудоемкость'].sum().sort_values(ascending=False)

    print("\nТоп-10 типов операций по общей трудоемкости (Срезы):")
    for op_type, workload in top_operations_srez.head(10).items():
        if pd.notna(op_type):
            print(f"{op_type} - {workload:.2f} трудоемкости")
else:
    print("DataFrame 'df_srez' пуст или отсутствуют колонки 'Название операции' или 'Общая трудоемкость'. Топ-10 типов операций по загруженности невозможно.")

# Анализ df_fakt
if not df_fakt.empty and 'Название операции' in df_fakt.columns and 'Закрытая трудоемкость' in df_fakt.columns:
    print("\n--- Анализ 'ОтметкаФакта.csv': Топ-10 типов операций по загруженности ---")
    # Группируем по названию операции и суммируем закрытую трудоемкость
    top_operations_fakt = df_fakt.groupby('Название операции')['Закрытая трудоемкость'].sum().sort_values(ascending=False)

    print("\nТоп-10 типов операций по закрытой трудоемкости (ОтметкаФакта):")
    for op_type, workload in top_operations_fakt.head(10).items():
        if pd.notna(op_type):
            print(f"{op_type} - {workload:.2f} трудоемкости")
else:
    print("DataFrame 'df_fakt' пуст или отсутствуют колонки 'Название операции' или 'Закрытая трудоемкость'. Топ-10 типов операций по загруженности невозможно.")

print("\n--- Анализ Топ-10 типов операций по загруженности завершен. ---")

# --- Обнаружение сезонности/циклов по датам ---

print("\n--- Обнаружение сезонности/циклов по датам ---")

# Анализ df_srez
if not df_srez.empty and 'Дата записи среза' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
    print("\n--- Анализ 'Срезы2024_2025.csv': Сезонность по общей трудоемкости ---")

    # Установим дату как индекс для временных рядов
    df_srez_time_series = df_srez.set_index('Дата записи среза').copy()

    # Агрегация по дням
    daily_srez_workload = df_srez_time_series['Общая трудоемкость'].resample('D').sum()
    print("\nЕжедневная динамика общей трудоемкости (Срезы) - первые 10 дней:")
    print(daily_srez_workload.head(10))
    print("\nЕжедневная динамика общей трудоемкости (Срезы) - последние 10 дней:")
    print(daily_srez_workload.tail(10))

    # Агрегация по неделям
    weekly_srez_workload = df_srez_time_series['Общая трудоемкость'].resample('W').sum()
    print("\nЕженедельная динамика общей трудоемкости (Срезы) - первые 10 недель:")
    print(weekly_srez_workload.head(10))
    print("\nЕженедельная динамика общей трудоемкости (Срезы) - последние 10 недель:")
    print(weekly_srez_workload.tail(10))

    # Агрегация по месяцам
    monthly_srez_workload = df_srez_time_series['Общая трудоемкость'].resample('ME').sum()
    print("\nЕжемесячная динамика общей трудоемкости (Срезы):")
    print(monthly_srez_workload)

    print("\nДля более точного выявления сезонности рекомендуется визуализировать эти данные (например, с помощью линейных графиков) и/или использовать статистические методы разложения временных рядов (например, STL decomposition).")
else:
    print("DataFrame 'df_srez' пуст или отсутствуют колонки 'Дата записи среза' или 'Общая трудоемкость'. Анализ сезонности невозможен.")

# Анализ df_fakt
if not df_fakt.empty and 'Дата закрытия операции' in df_fakt.columns and 'Закрытая трудоемкость' in df_fakt.columns:
    print("\n--- Анализ 'ОтметкаФакта.csv': Сезонность по закрытой трудоемкости ---")

    # Установим дату как индекс для временных рядов
    df_fakt_time_series = df_fakt.set_index('Дата закрытия операции').copy()

    # Агрегация по дням
    daily_fakt_workload = df_fakt_time_series['Закрытая трудоемкость'].resample('D').sum()
    print("\nЕжедневная динамика закрытой трудоемкости (ОтметкаФакта) - первые 10 дней:")
    print(daily_fakt_workload.head(10))
    print("\nЕжедневная динамика закрытой трудоемкости (ОтметкаФакта) - последние 10 дней:")
    print(daily_fakt_workload.tail(10))

    # Агрегация по неделям
    weekly_fakt_workload = df_fakt_time_series['Закрытая трудоемкость'].resample('W').sum()
    print("\nЕженедельная динамика закрытой трудоемкости (ОтметкаФакта) - первые 10 недель:")
    print(weekly_fakt_workload.head(10))
    print("\nЕженедельная динамика закрытой трудоемкости (ОтметкаФакта) - последние 10 недель:")
    print(weekly_fakt_workload.tail(10))

    # Агрегация по месяцам
    monthly_fakt_workload = df_fakt_time_series['Закрытая трудоемкость'].resample('ME').sum()
    print("\nЕжемесячная динамика закрытой трудоемкости (ОтметкаФакта):")
    print(monthly_fakt_workload)

    print("\nДля более точного выявления сезонности рекомендуется визуализировать эти данные (например, с помощью линейных графиков) и/или использовать статистические методы разложения временных рядов (например, STL decomposition).")
else:
    print("DataFrame 'df_fakt' пуст или отсутствуют колонки 'Дата закрытия операции' или 'Закрытая трудоемкость'. Анализ сезонности невозможен.")

print("\n--- Анализ сезонности/циклов завершен. ---")


# --- Средняя и медианная загрузка по участкам и ШГО ---
print("\n--- Расчет средней и медианной загрузки по участкам и ШГО ---")

# Анализ df_srez (плановая загрузка)
if not df_srez.empty:
    print("\n--- Анализ 'Срезы2024_2025.csv' (Плановая загрузка) ---")

    # Средняя и медианная загрузка по участкам
    if 'main_department' in df_srez.columns and 'sub_department' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
        print("\nСредняя и медианная загрузка по участкам (Срезы):")
        # Группируем по основному подразделению и участку, затем считаем среднее и медиану общей трудоемкости
        avg_median_workload_srez_dept = df_srez.dropna(subset=['main_department', 'sub_department', 'Общая трудоемкость']).groupby(['main_department', 'sub_department'])['Общая трудоемкость'].agg(['mean', 'median']).sort_values(by='mean', ascending=False)
        print(avg_median_workload_srez_dept.to_string())
    else:
        print("Недостаточно данных для расчета средней и медианной загрузки по участкам в df_srez. Убедитесь, что колонки 'main_department', 'sub_department' и 'Общая трудоемкость' существуют и не пусты.")

    # Средняя и медианная загрузка по ШГО
    if 'ШГО – вид оборудования' in df_srez.columns and 'Общая трудоемкость' in df_srez.columns:
        print("\nСредняя и медианная загрузка по ШГО (Срезы):")
        # Группируем по ШГО, затем считаем среднее и медиану общей трудоемкости
        avg_median_workload_srez_shgo = df_srez.dropna(subset=['ШГО – вид оборудования', 'Общая трудоемкость']).groupby('ШГО – вид оборудования')['Общая трудоемкость'].agg(['mean', 'median']).sort_values(by='mean', ascending=False)
        print(avg_median_workload_srez_shgo.to_string())
    else:
        print("Недостаточно данных для расчета средней и медианной загрузки по ШГО в df_srez. Убедитесь, что колонки 'ШГО – вид оборудования' и 'Общая трудоемкость' существуют и не пусты.")
else:
    print("DataFrame 'df_srez' пуст. Расчет средней и медианной загрузки невозможен.")


# Анализ df_fakt (фактическая загрузка)
if not df_fakt.empty:
    print("\n--- Анализ 'ОтметкаФакта.csv' (Фактическая загрузка) ---")

    # Средняя и медианная загрузка по участкам
    if 'main_department' in df_fakt.columns and 'sub_department' in df_fakt.columns and 'Закрытая трудоемкость' in df_fakt.columns:
        print("\nСредняя и медианная загрузка по участкам (ОтметкаФакта):")
        # Группируем по основному подразделению и участку, затем считаем среднее и медиану закрытой трудоемкости
        avg_median_workload_fakt_dept = df_fakt.dropna(subset=['main_department', 'sub_department', 'Закрытая трудоемкость']).groupby(['main_department', 'sub_department'])['Закрытая трудоемкость'].agg(['mean', 'median']).sort_values(by='mean', ascending=False)
        print(avg_median_workload_fakt_dept.to_string())
    else:
        print("Недостаточно данных для расчета средней и медианной загрузки по участкам в df_fakt. Убедитесь, что колонки 'main_department', 'sub_department' и 'Закрытая трудоемкость' существуют и не пусты.")

    # Средняя и медианная загрузка по ШГО
    if 'ШГО – вид оборудования' in df_fakt.columns and 'Закрытая трудоемкость' in df_fakt.columns:
        print("\nСредняя и медианная загрузка по ШГО (ОтметкаФакта):")
        # Группируем по ШГО, затем считаем среднее и медиану закрытой трудоемкости
        avg_median_workload_fakt_shgo = df_fakt.dropna(subset=['ШГО – вид оборудования', 'Закрытая трудоемкость']).groupby('ШГО – вид оборудования')['Закрытая трудоемкость'].agg(['mean', 'median']).sort_values(by='mean', ascending=False)
        print(avg_median_workload_fakt_shgo.to_string())
    else:
        print("Недостаточно данных для расчета средней и медианной загрузки по ШГО в df_fakt. Убедитесь, что колонки 'ШГО – вид оборудования' и 'Закрытая трудоемкость' существуют и не пусты.")
else:
    print("DataFrame 'df_fakt' пуст. Расчет средней и медианной загрузки невозможен.")

print("\n--- Расчет средней и медианной загрузки завершен. ---")


## Обнаружение максимальных задержек операций

print("\n--- Обнаружение максимальных задержек операций ---")

# Ключевые колонки для объединения
merge_cols = [
    'Номер заказа на производство',
    'Номер операции',
    'Название операции',
    'Артикул номенклатуры' # Добавлено для более точного соответствия операции
]

# Проверяем существование исходных колонок, из которых будут формироваться суффиксы
required_srez_date_col = 'Дата и время начала операции'
required_fakt_date_col = 'Дата закрытия операции'

missing_srez_cols_for_merge = [col for col in merge_cols + [required_srez_date_col] if col not in df_srez.columns]
missing_fakt_cols_for_merge = [col for col in merge_cols + [required_fakt_date_col] if col not in df_fakt.columns]


if missing_srez_cols_for_merge:
    print(f"Внимание: В df_srez отсутствуют необходимые колонки для анализа задержек: {missing_srez_cols_for_merge}. Пропускаем анализ задержек.")
    merged_delays = pd.DataFrame()
elif missing_fakt_cols_for_merge:
    print(f"Внимание: В df_fakt отсутствуют необходимые колонки для анализа задержек: {missing_fakt_cols_for_merge}. Пропускаем анализ задержек.")
    merged_delays = pd.DataFrame()
else:
    # Важно: делаем копии, чтобы избежать SettingWithCopyWarning при дальнейших операциях
    df_srez_for_merge = df_srez.copy()
    df_fakt_for_merge = df_fakt.copy()

    # Объединяем df_srez и df_fakt для нахождения задержек
    # Используем inner join, чтобы получить только операции, которые есть и в плане, и в факте
    # Добавляем суффиксы, чтобы различать одноименные колонки
    merged_delays = pd.merge(
        df_srez_for_merge,
        df_fakt_for_merge,
        on=merge_cols,
        how='inner',
        suffixes=('_plan', '_fact')
    )
    print(f"Количество строк после объединения для анализа задержек: {len(merged_delays)}")

    # --- ОТЛАДОЧНЫЕ ПРИНТЫ: Проверяем столбцы сразу после объединения ---
    print("\n--- ДЕБАГ: Столбцы в merged_delays после объединения ---")
    print(merged_delays.columns.tolist())
    print("--- КОНЕЦ ДЕБАГА ---")

    if not merged_delays.empty:
        # Проверяем, что колонки с суффиксами существуют после объединения
        if 'Дата и время начала операции_plan' in merged_delays.columns and \
           'Дата закрытия операции_fact' in merged_delays.columns:

            # Преобразуем даты в datetime. errors='coerce' заменит некорректные даты на NaT
            merged_delays['Дата и время начала операции_plan'] = pd.to_datetime(merged_delays['Дата и время начала операции_plan'], errors='coerce')
            merged_delays['Дата закрытия операции_fact'] = pd.to_datetime(merged_delays['Дата закрытия операции_fact'], errors='coerce')

            # Удаляем строки, где даты не удалось преобразовать (стали NaT)
            merged_delays.dropna(subset=['Дата и время начала операции_plan', 'Дата закрытия операции_fact'], inplace=True)

            if not merged_delays.empty:
                # Рассчитываем задержку в днях
                # Если фактическая дата закрытия позже плановой даты начала, то это задержка
                merged_delays['Задержка в днях'] = (merged_delays['Дата закрытия операции_fact'] - merged_delays['Дата и время начала операции_plan']).dt.days

                # Фильтруем только положительные задержки (т.е. когда факт позже плана)
                delayed_operations = merged_delays[merged_delays['Задержка в днях'] > 0].copy()

                if not delayed_operations.empty:
                    # Сортируем по убыванию задержки
                    delayed_operations_sorted = delayed_operations.sort_values(by='Задержка в днях', ascending=False)

                    print("\nТоп-20 операций с максимальными задержками (разница между фактическим закрытием и плановым началом):")
                    # Выводим только самые важные колонки для краткости
                    print(delayed_operations_sorted[[
                        'Номер заказа на производство',
                        'Номер операции',
                        'Название операции_plan',
                        'Артикул номенклатуры',
                        'Дата и время начала операции_plan',
                        'Дата закрытия операции_fact',
                        'Задержка в днях'
                    ]].head(20).to_string())
                else:
                    print("Операций с положительными задержками не обнаружено.")
            else:
                print("Нет данных для анализа задержек после удаления строк с некорректными датами.")
        else:
            print("Одна или обе колонки дат с суффиксами ('Дата и время начала операции_plan' или 'Дата закрытия операции_fact') не найдены в merged_delays после объединения.")
    else:
        print("Не удалось объединить 'df_srez' и 'df_fakt' по общим операциям. Проверьте данные и ключи для объединения.")



## Анализ тенденции сдвига даты начала операции по срезам плана

print("\n--- Анализ тенденции сдвига даты начала операции по срезам плана ---")

if not df_srez.empty and \
   'Номер заказа на производство' in df_srez.columns and \
   'Номер операции' in df_srez.columns and \
   'Дата записи среза' in df_srez.columns and \
   'Дата и время начала операции' in df_srez.columns and \
   'Название операции' in df_srez.columns and \
   'Артикул номенклатуры' in df_srez.columns:

    print("Начало анализа сдвигов дат начала операций (оптимизированный подход)...")

    # Сортировка данных по уникальному идентификатору операции и дате среза
    df_srez_sorted = df_srez.sort_values(by=[
        'Номер заказа на производство',
        'Номер операции',
        'Артикул номенклатуры',
        'Дата записи среза'
    ]).copy()

    # Создание уникального ID для каждой операции
    # Этот ID должен быть максимально уникальным, чтобы корректно отслеживать одну и ту же операцию
    df_srez_sorted['operation_id'] = df_srez_sorted['Номер заказа на производство'].astype(str) + '_' + \
                                      df_srez_sorted['Номер операции'].astype(str) + '_' + \
                                      df_srez_sorted['Артикул номенклатуры'].astype(str)

    # Группировка по operation_id и применение lambda-функции для получения предыдущей даты среза
    # и предыдущей даты начала операции. shift(1) берет значение из предыдущей строки в группе.
    df_srez_sorted['prev_slice_date'] = df_srez_sorted.groupby('operation_id')['Дата записи среза'].shift(1)
    df_srez_sorted['prev_start_date'] = df_srez_sorted.groupby('operation_id')['Дата и время начала операции'].shift(1)

    # Фильтруем строки, где есть предыдущие значения для сравнения (т.е. где был хотя бы один предыдущий срез для этой операции)
    shifts_df = df_srez_sorted.dropna(subset=['prev_slice_date', 'prev_start_date']).copy()

    if not shifts_df.empty:
        # Рассчитываем сдвиг в днях
        # Сдвиг = текущая_дата_начала - предыдущая_дата_начала
        shifts_df['Сдвиг в днях'] = (shifts_df['Дата и время начала операции'] - shifts_df['prev_start_date']).dt.days

        # Определяем направление сдвига
        shifts_df['Направление сдвига'] = shifts_df['Сдвиг в днях'].apply(
            lambda x: 'Опережение' if x < 0 else ('Задержка' if x > 0 else 'Без сдвига')
        )

        print("\nОбнаружены сдвиги даты начала операций (первые 20 строк):")
        # Выводим только самые важные колонки для краткости
        print(shifts_df[[
            'Номер заказа на производство',
            'Номер операции',
            'Название операции',
            'Артикул номенклатуры',
            'Дата записи среза',
            'prev_slice_date',
            'Дата и время начала операции',
            'prev_start_date',
            'Сдвиг в днях',
            'Направление сдвига'
        ]].head(20).to_string()) # Используем .head(20) для вывода части, а не всего

        # Анализ общей тенденции: средний сдвиг
        avg_shift = shifts_df['Сдвиг в днях'].mean()
        median_shift = shifts_df['Сдвиг в днях'].median()
        print(f"\nСредний сдвиг даты начала операции: {avg_shift:.2f} дней")
        print(f"Медианный сдвиг даты начала операции: {median_shift:.2f} дней")

        # Распределение сдвигов
        shift_distribution = shifts_df['Направление сдвига'].value_counts(normalize=True) * 100
        print("\nРаспределение сдвигов:")
        print(shift_distribution.to_string())

        # Топ-10 операций по абсолютной величине сдвига
        # Используем .loc с .nlargest для получения строк с наибольшими абсолютными сдвигами
        top_shifted_ops = shifts_df.loc[shifts_df['Сдвиг в днях'].abs().nlargest(10).index]
        print("\nТоп-10 операций с наибольшим абсолютным сдвигом:")
        print(top_shifted_ops[[
            'Номер заказа на производство',
            'Номер операции',
            'Название операции',
            'Артикул номенклатуры',
            'Дата записи среза',
            'prev_slice_date',
            'Дата и время начала операции',
            'prev_start_date',
            'Сдвиг в днях',
            'Направление сдвига'
        ]].to_string())

    else:
        print("Недостаточно данных в df_srez для анализа сдвигов дат начала операций после фильтрации. Убедитесь, что для каждой операции есть несколько срезов.")
else:
    print("DataFrame 'df_srez' пуст или отсутствуют необходимые колонки для анализа сдвигов (например, 'Номер заказа на производство', 'Номер операции', 'Дата записи среза', 'Дата и время начала операции', 'Название операции', 'Артикул номенклатуры').")

print("\n--- Анализ сдвигов и задержек завершен. ---")