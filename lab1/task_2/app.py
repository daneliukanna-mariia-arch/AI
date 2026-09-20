import io
import numpy as np
import panel as pn
from PIL import Image, ImageDraw

# Ініціалізація розширень Panel
pn.extension()

GRID_ROWS = 5
GRID_COLS = 5


def calculate_feature_vector(image):
  image = image.convert("L")
  width, height = image.size
  pixels = np.array(image)
  cell_width = width / GRID_COLS
  cell_height = height / GRID_ROWS
  absolute_vector = []

  for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
      x1 = int(col * cell_width)
      x2 = int((col + 1) * cell_width)
      y1 = int(row * cell_height)
      y2 = int((row + 1) * cell_height)

      cell = pixels[y1:y2, x1:x2]
      black_pixels = np.sum(cell < 128)
      absolute_vector.append(int(black_pixels))

  total_black_pixels = sum(absolute_vector)

  if total_black_pixels > 0:
    normalized_vector = [
        round(value / total_black_pixels, 4) for value in absolute_vector
    ]
  else:
    normalized_vector = [0.0] * len(absolute_vector)

  grid_image = image.convert("RGB").copy()
  draw = ImageDraw.Draw(grid_image)

  for col in range(1, GRID_COLS):
    x = int(col * cell_width)
    draw.line((x, 0, x, height), fill="red", width=2)

  for row in range(1, GRID_ROWS):
    y = int(row * cell_height)
    draw.line((0, y, width, y), fill="red", width=2)

  return absolute_vector, normalized_vector, grid_image


def calculate_distance(v1, v2, norm_type):
  v1 = np.array(v1)
  v2 = np.array(v2)

  if norm_type == "Евклідова норма":
    # Евклідова відстань: sqrt(sum((x1 - x2)^2))
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))
  elif norm_type == "Норма Чебишева":
    # Чебишевська відстань: max(|x1 - x2|)
    return float(np.max(np.abs(v1 - v2)))
  elif norm_type == "Манхетенська норма":
    # Манхетенська відстань: sum(|x1 - x2|)
    return float(np.sum(np.abs(v1 - v2)))
  return 0.0


# --- Елементи інтерфейсу для 3 еталонів ---
file_input_1 = pn.widgets.FileInput(
    accept=".bmp,image/*", name="Еталон 1 (Клас 1)"
)
file_input_2 = pn.widgets.FileInput(
    accept=".bmp,image/*", name="Еталон 2 (Клас 2)"
)
file_input_3 = pn.widgets.FileInput(
    accept=".bmp,image/*", name="Еталон 3 (Клас 3)"
)

pane_1 = pn.pane.Image(sizing_mode="scale_width", width=120)
pane_2 = pn.pane.Image(sizing_mode="scale_width", width=120)
pane_3 = pn.pane.Image(sizing_mode="scale_width", width=120)

out_1 = pn.widgets.TextAreaInput(name="Вектори Еталона 1", rows=6, disabled=True)
out_2 = pn.widgets.TextAreaInput(name="Вектори Еталона 2", rows=6, disabled=True)
out_3 = pn.widgets.TextAreaInput(name="Вектори Еталона 3", rows=6, disabled=True)

# --- Елементи інтерфейсу для невідомого образу ---
file_input_unk = pn.widgets.FileInput(
    accept=".bmp,image/*", name="Невідомий образ"
)
pane_unk = pn.pane.Image(sizing_mode="scale_width", width=150)
grid_pane_unk = pn.pane.Image(sizing_mode="scale_width", width=150)
out_unk = pn.widgets.TextAreaInput(
    name="Вектори невідомого образу", rows=8, disabled=True
)

# --- Налаштування класифікації ---
norm_select = pn.widgets.Select(
    name="Варіант норми",
    options=["Евклідова норма", "Норма Чебишева", "Манхетенська норма"],
)
classify_button = pn.widgets.Button(
    name="Класифікувати образ", button_type="primary"
)
result_output = pn.widgets.TextAreaInput(
    name="Результати порівняння та класифікації", rows=10, sizing_mode="stretch_width", disabled=True
)


# --- Логіка обробки та класифікації ---
def on_classify(event):
  if not (
      file_input_1.value
      and file_input_2.value
      and file_input_3.value
      and file_input_unk.value
  ):
    result_output.value = (
        "Помилка: Завантажте всі 3 еталони та невідоме зображення!"
    )
    return

  try:
    # 1. Еталон 1
    img1 = Image.open(io.BytesIO(file_input_1.value))
    pane_1.object = img1
    abs1, norm1, _ = calculate_feature_vector(img1)
    out_1.value = f"Абс: {abs1}\n\nНорм: {norm1}"

    # 2. Еталон 2
    img2 = Image.open(io.BytesIO(file_input_2.value))
    pane_2.object = img2
    abs2, norm2, _ = calculate_feature_vector(img2)
    out_2.value = f"Абс: {abs2}\n\nНорм: {norm2}"

    # 3. Еталон 3
    img3 = Image.open(io.BytesIO(file_input_3.value))
    pane_3.object = img3
    abs3, norm3, _ = calculate_feature_vector(img3)
    out_3.value = f"Абс: {abs3}\n\nНорм: {norm3}"

    # 4. Невідомий образ
    img_unk = Image.open(io.BytesIO(file_input_unk.value))
    pane_unk.object = img_unk
    abs_unk, norm_unk, grid_unk = calculate_feature_vector(img_unk)
    grid_pane_unk.object = grid_unk
    out_unk.value = (
        f"Абсолютний вектор:\n{abs_unk}\n\nНормований вектор:\n{norm_unk}"
    )

    # 5. Розрахунок відстаней за обраною нормою (використовуємо нормовані вектори)
    norm_type = norm_select.value
    d1 = calculate_distance(norm_unk, norm1, norm_type)
    d2 = calculate_distance(norm_unk, norm2, norm_type)
    d3 = calculate_distance(norm_unk, norm3, norm_type)

    distances = {"Клас 1": d1, "Клас 2": d2, "Клас 3": d3}
    best_class = min(distances, key=distances.get)

    # 6. Формування результату
    res_text = f"Обраний метод: {norm_type}\n\nМіри відповідності (відстані):\n"
    for cls_name, dist in distances.items():
      res_text += f" • {cls_name}: {dist:.4f}\n"

    res_text += f"\nВисновок: Невідомий образ належить до **{best_class}**"
    result_output.value = res_text

  except Exception as e:
    result_output.value = f"Помилка виконання обчислень: {e}"


classify_button.on_click(on_classify)

# --- Побудова макету сторінки ---
layout = pn.Column(
    pn.pane.Markdown(
        """
        # Система розпізнавання образів за методом порівняння з еталоном
        Завантажте **3 еталонні зображення** для відповідних класів, а також **невідомий образ**. 
        Оберіть бажаний тип норми для розрахунку відстаней та натисніть кнопку класифікації.
        """
    ),
    pn.pane.Markdown("## 1. Еталонні образи"),
    pn.Row(
        pn.Column(file_input_1, pane_1, out_1),
        pn.Column(file_input_2, pane_2, out_2),
        pn.Column(file_input_3, pane_3, out_3),
    ),
    pn.layout.Divider(),
    pn.pane.Markdown("## 2. Невідомий образ та класифікація"),
    pn.Row(
        pn.Column(
            file_input_unk,
            pn.Row(pane_unk, grid_pane_unk),
            out_unk,
            width=400,
        ),
        pn.Column(
            norm_select, classify_button, result_output, sizing_mode="stretch_width"
        ),
    ),
    sizing_mode="stretch_width",
)

# Запуск додатку
layout.servable()
