import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from pypdf import PdfReader




load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Моделі
MODEL_FLASH_36 = "gemini-3.6-flash"
MODEL_FLASH_25 = "gemini-2.5-flash"

# Текст про квантову запутаність
TEXT_QUANTUM = """
Фізики впервые розробили методику прямого вимірювання складних W-станів квантової запутаності, подолавши бар'єр, що існував понад чверть століття.
Квантова запутаність залишається одним із найбільш вражаючих і контрінтуїтивних явищ сучасної фізики. Коли фундаментальні елементарні частинки утворюють запутану систему, їхній стан стає єдиним цілим.
Для проведення складного експерименту вченим довелося подолати класичні перешкоди квантової томографії, яка потребує величезної кількості ідентичних систем і незліченних вимірювань.

Надіслані крізь нього фотони взаємодіяли завдяки інтерференції хвиль, що дозволило зафіксувати циклічну симетрію зсуву системи без її поетапного руйнування.
Отримані результати продемонстрували середню точність розпізнавання на рівні приблизно 87 відсотків, що впевнено перевищило теоретичний поріг і підтвердило життєздатність обраної методики.
"""

# Допоміжна функція безпечного виклику API
def safe_generate_content(prompt_or_contents, model=MODEL_FLASH_36, config=None, max_retries=5, initial_delay=3):
    """Безпечний виклик API з обробкою перевантаження (503) та лімітів (429)"""
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt_or_contents,
                config=config
            )
        except (ServerError, APIError, ClientError) as e:
            if attempt == max_retries:
                print(f"Помилка після {max_retries} спроб: {e}")
                raise e
            print(f"Помилка сервера ({type(e).__name__}). Повторна спроба {attempt}/{max_retries} через {delay} сек...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"Помилка: {e}")
            raise e

# Розбиття на чанки (Map-Reduce)
def create_chunks(text, chunk_size=1500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Sliding Window chunks
def create_sliding_window_chunks(text, window_size=10000, overlap_size=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(words):
        word = words[i]
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= window_size:
            chunks.append(" ".join(current_chunk))
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_chunk):
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
                if overlap_len >= overlap_size:
                    break
            current_chunk = overlap_words
            current_length = overlap_len
        i += 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks



# ЗАВДАННЯ 1. Використання просунутих технік Prompt Engineering

def task_1():
    print("\nЗАВДАННЯ 1: PROMPT ENGINEERING")
    
    # 1.1. Звичайний prompt
    prompt_ordinary = f"Проаналізуй наведений текст та сформуй короткий висновок.\n\n{TEXT_QUANTUM}"
    resp1 = safe_generate_content(prompt_ordinary, model=MODEL_FLASH_36)
    print("\n--- 1.1. Звичайний промпт ---")
    print(resp1.text)
    
    # 1.2. Просунутий prompt
    prompt_improved = f"""[Роль моделі]
Ти – провідний експерт із квантової фізики та науковий аналітик із досвідом висвітлення складних наукових відкриттів для професійної аудиторії.
[Контекст]
Я надаю тобі статтю про прорив дослідників із Японії у сфері квантової запутаності (пряме вимірювання W-станів для трьох фотонів).
[Чітка інструкція]
Проаналізуй наданий текст та сформуй структурований науково-аналітичний висновок про значення цього дослідження для квантової фізики та практичних технологій.
[Декомпозиція задачі на етапи]
Виконай аналіз послідовно за такими кроками:
1. Основне досягнення: Виділи головну наукову новизну дослідження (що саме зробили вчені та які обмеження подолали).
2. Технічна суть: Поясни використаний метод і в чому полягає унікальність W-станів порівняно зі станами GHZ.
3. Результати та перспективи: Вкажіть точність методу та ключові напрямки практичного застосування у майбутньому.
4. Підсумковий висновок: Сформулюй підсумок (1-2 речення), який пояснює значення цієї події для квантових технологій.
[Обмеження]
Обсяг відповіді – до 150-200 слів.
Використовуй тільки факти з наданого тексту (не додавай зовнішньої інформації).
Складні квантові поняття описуй лаконічно та зрозуміло.
Дотримуйся ділового/наукового тону мовлення.
[Формат результату]
Оформи відповідь у такому вигляді:
  Ключовий прорив: [короткий опис]
  Метод та особливості: [короткий опис]
  Результати та застосування: [короткий опис]
  Підсумок: [1-2 речення]

Вхідний текст для аналізу:
{TEXT_QUANTUM}
"""
    resp2 = safe_generate_content(prompt_improved, model=MODEL_FLASH_36)
    print("\n--- 1.2. Просунутий промпт ---")
    print(resp2.text)



# ЗАВДАННЯ 2. Декомпозиція складної задачі

def task_2():
    print("\n ЗАВДАННЯ 2: ДЕКОМПОЗИЦІЯ ЗАДАЧІ ")
    
    prompt_simple_tests = f"Згенеруй тестові питання до тексту:\n\n{TEXT_QUANTUM}"
    resp_a = safe_generate_content(prompt_simple_tests, model=MODEL_FLASH_36)
    print("\n--- 2.1. Пряма генерація тестів ---")
    print(resp_a.text)
    
    prompt_decomp = f"""1. визначити основні поняття тексту;
2. визначити ключові факти;
3. сформувати запитання;
4. сформувати варіанти відповідей;
5. визначити правильну відповідь.

{TEXT_QUANTUM}
"""
    resp_b = safe_generate_content(prompt_decomp, model=MODEL_FLASH_36)
    print("\n--- 2.2. Генерація тестів через декомпозицію ---")
    print(resp_b.text)



# ЗАВДАННЯ 3. Імітація ReAct

def task_3():
    print("\n ЗАВДАННЯ 3: ІМІТАЦІЯ ReAct ")
    prompt_react = f"""Проаналізуй наданий текст та класифікуй викладену в ньому інформацію, розділивши її на об'єктивні факти (верифіковані дані, вимірювання, події) та суб'єктивні висновки/інтерпретації автора (оцінки, припущення, прогнози).
[Послідовність виконання]
Виконай завдання суворо за такими 4 кроками:
1. План дій: Коротко розпиши алгоритм (3-4 пункти), за яким ти здійснюватимеш аналіз тексту.
2. Виконання: Розбий текст на смислові тези та виділи з них факти та висновки із цитуванням або посиланням на фрагменти.
3. Перевірка: Проведіть самоперевірку за критеріями: чи немає серед «фактів» прихованих оцінок автора, і чи не потрапили об'єктивні дані до «висновків».
4. Результат: Подай підсумкову таблицю та підбий короткий аналітичний підсумок (2-3 речення).

{TEXT_QUANTUM}
"""
    resp = safe_generate_content(prompt_react, model=MODEL_FLASH_36)
    print("\n--- 3. Імітація ReAct ---")
    print(resp.text)



# ЗАВДАННЯ 4. Формування структурованого результату (JSON)

def task_4():
    print("\n ЗАВДАННЯ 4: СТРУКТУРОВАНИЙ JSON")
    prompt_json = f"""Проаналізуй наданий текст про наукове відкриття у галузі квантової фізики. Твій результат має бути чітко серіалізованим JSON-об'єктом, який можна одразу використовувати.
[Обмеження та вимоги до формату]
1. Відповідь має містити ВИКЛЮЧНО валідний JSON-код (без вступних слів, привітань чи розмовного тексту).
2. Дотримуйся заданого схематичного іменування.
3. Усі виділені факти та висновки мають спиратися тільки на наданий текст.
[Структура JSON (JSON Schema)]
{{
  "article_metadata": {{
    "topic": "Назва або головна тема статті",
    "journal": "Назва журналу (якщо є)",
    "researchers": ["Університет/Організація 1", "Університет/Організація 2"]
  }},
  "scientific_breakthrough": {{
    "achievement": "Суть головного досягнення",
    "qubits_or_photons_count": "Кількість частинок у досліді",
    "measured_state": "Тип квантового стану",
    "accuracy_percentage": "Точність у відсотках (число або рядок)"
  }},
  "analysis": {{
    "objective_facts": [
      "Факт 1",
      "Факт 2"
    ],
    "author_conclusions_and_interpretations": [
      "Суб'єктивна оцінка або прогноз 1",
      "Суб'єктивна оцінка або прогноз 2"
    ]
  }},
  "future_applications": [
    "Сфера застосування 1",
    "Сфера застосування 2"
  ],
  "summary": "Підсумковий висновок 1-2 реченнями"
}}

{TEXT_QUANTUM}
"""
    resp = safe_generate_content(prompt_json, model=MODEL_FLASH_36)
    print("\n--- 4. Результат у форматі JSON ---")
    print(resp.text)



# ЗАВДАННЯ 5. Робота з великим контекстом

def task_5(pdf_path="elevator.pdf"):
    print("\n ЗАВДАННЯ 5: ВЕЛИКИЙ КОНТЕКСТ ТА CHUNKING ")
    
    if not os.path.exists(pdf_path):
        print(f"Файл {pdf_path} не знайдено.")
        return

    # 5.1. Пряма передача PDF через File API
    print("\n--- 5.1. Обробка PDF без Chunking ---")
    uploaded_file = client.files.upload(file=pdf_path)
    prompt_direct = "Проаналізуй наданий PDF-документ та сформуй структурований звіт: Ключові ідеї та основна суть документа."
    resp_pdf = safe_generate_content([uploaded_file, prompt_direct], model=MODEL_FLASH_36)
    print(resp_pdf.text)

    # Зчитування тексту з PDF
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"

    # 5.2. Map-Reduce Chunking
    print("\n--- 5.2. Map-Reduce Chunking ---")
    chunks = create_chunks(full_text, chunk_size=1500)
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        prompt_chunk = f"Зроби коротке узагальнення (3-4 тези) цього фрагмента:\n\n{chunk}"
        resp_c = safe_generate_content(prompt_chunk, model=MODEL_FLASH_25)
        if resp_c and resp_c.text:
            summaries.append(f"Узагальнення S{i}:\n{resp_c.text.strip()}")
    
    combined_summaries = "\n\n".join(summaries)
    final_prompt = f"""На основі поданих проміжних узагальнень з усіх частин документа сформуй єдиний підсумковий звіт:

{combined_summaries}

Формат результату:
1. Головна ідея документа.
2. Ключові тези.
3. Підсумкові висновки.
"""
    final_resp = safe_generate_content(final_prompt, model=MODEL_FLASH_25)
    print(final_resp.text)

    # 5.3. Sliding Window
    print("\n--- 5.3. Sliding Window ---")
    sw_chunks = create_sliding_window_chunks(full_text, window_size=10000, overlap_size=1000)
    sw_summaries = []
    for i, chunk in enumerate(sw_chunks, 1):
        prompt_sw = f"Зроби коротке узагальнення (3-4 тези) цього фрагмента:\n\n{chunk}"
        resp_sw = safe_generate_content(prompt_sw, model=MODEL_FLASH_36)
        if resp_sw and resp_sw.text:
            sw_summaries.append(f"Узагальнення S{i}:\n{resp_sw.text.strip()}")

    combined_sw = "\n\n".join(sw_summaries)
    final_sw_prompt = f"""На основі поданих проміжних узагальнень з усіх частин документа сформуй єдиний підсумковий звіт.
Усунь повтори, якщо однакові тези потрапили в сусідні фрагменти через перекриття вікон.

{combined_sw}

Формат результату:
1. Головна ідея документа.
2. Ключові тези.
3. Підсумкові висновки.
"""
    final_sw_resp = safe_generate_content(final_sw_prompt, model=MODEL_FLASH_36)
    print(final_sw_resp.text)



# ЗАВДАННЯ 6. Дослідження параметрів генерації (Temperature & Top-P)

def task_6():
    print("\nЗАВДАННЯ 6: ТЕМПЕРАТУРА ТА TOP-P ")
    
    prompt_ideas = (
        "Запропонуй 5 креативних та практичних ідей, як можна використовувати "
        "генеративний ШІ для автоматизації щоденної рутинної роботи під час навчання чи розробки ПЗ."
    )

    # 6.1. Temperature
    print("\n--- 6.1. Дослідження Temperature ---")
    for temp in [0.0, 0.5, 1.0]:
        print(f"\n=== ГЕНЕРАЦІЯ ДЛЯ ТЕМПЕРАТУРИ: {temp} ===")
        cfg = types.GenerateContentConfig(temperature=temp)
        res = safe_generate_content(prompt_ideas, model=MODEL_FLASH_36, config=cfg)
        print(res.text[:400] + "\n...")

    # 6.2. Top-P
    print("\n--- 6.2. Дослідження Top-P ---")
    for top_p in [0.5, 0.8, 1.0]:
        print(f"\n=== ГЕНЕРАЦІЯ ДЛЯ TOP_P: {top_p} (temp=0.7) ===")
        cfg = types.GenerateContentConfig(temperature=0.7, top_p=top_p)
        res = safe_generate_content(prompt_ideas, model=MODEL_FLASH_36, config=cfg)
        print(res.text[:400] + "\n...")



# Точка входу

if __name__ == "__main__":
    task_1()
    task_2()
    task_3()
    task_4()
    if os.path.exists("elevator.pdf"):
        task_5("elevator.pdf")
    task_6()
