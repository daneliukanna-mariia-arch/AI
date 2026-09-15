import io
import panel as pn
from PIL import Image, ImageDraw
import numpy as np

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
            round(value / total_black_pixels, 4)
            for value in absolute_vector
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

    absolute_text = "Абсолютний вектор ознак:\n\n" + str(absolute_vector)
    normalized_text = "Нормований вектор ознак:\n\n" + str(normalized_vector)

    return absolute_text, normalized_text, grid_image


# Створення елементів інтерфейсу (віджетів)
file_input = pn.widgets.FileInput(accept=".bmp,image/*", name="Завантажте BMP-зображення")
calc_button = pn.widgets.Button(name="Побудувати вектор ознак", button_type="primary")

input_image_pane = pn.pane.Image(sizing_mode="scale_width", width=300)
grid_output_pane = pn.pane.Image(sizing_mode="scale_width", width=300)

absolute_output = pn.widgets.TextAreaInput(name="Абсолютний вектор ознак", rows=8, disabled=True)
normalized_output = pn.widgets.TextAreaInput(name="Нормований вектор ознак", rows=8, disabled=True)


# Логіка обробки натискання кнопки
def on_click(event):
    if not file_input.value:
        absolute_output.value = "Зображення не завантажено."
        normalized_output.value = ""
        grid_output_pane.object = None
        input_image_pane.object = None
        return

    try:
        image = Image.open(io.BytesIO(file_input.value))
        input_image_pane.object = image
        
        abs_text, norm_text, grid_img = calculate_feature_vector(image)
        absolute_output.value = abs_text
        normalized_output.value = norm_text
        grid_output_pane.object = grid_img
    except Exception as e:
        absolute_output.value = f"Помилка обробки зображення: {e}"
        normalized_output.value = ""
        grid_output_pane.object = None


calc_button.on_click(on_click)

# Побудова макету сторінки
layout = pn.Column(
    pn.pane.Markdown(
        """
        # Система побудови вектора ознак

        Завантажте чорно-біле зображення у форматі BMP.
        
        Програма розділить його на сітку 5×5,
        порахує кількість чорних пікселів у кожній області
        та сформує абсолютний і нормований вектори ознак.
        """
    ),
    pn.Row(
        pn.Column(file_input, calc_button, input_image_pane),
        pn.Column(grid_output_pane)
    ),
    pn.pane.Markdown("## Результати"),
    pn.Row(
        absolute_output,
        normalized_output
    ),
    sizing_mode="stretch_width"
)

# Запуск додатку
layout.servable()
