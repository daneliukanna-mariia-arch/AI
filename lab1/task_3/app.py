import panel as pn
from PIL import Image, ImageDraw
import numpy as np
import io

pn.extension()

ROWS = 4
COLS = 5


def get_features(img):
    if img is None:
        return None, None, None

    gray = img.convert("L")
    width, height = gray.size
    data = np.array(gray)

    part_width = width / COLS
    part_height = height / ROWS

    vector = []

    for r in range(ROWS):
        for c in range(COLS):
            left = int(c * part_width)
            right = int((c + 1) * part_width)
            top = int(r * part_height)
            bottom = int((r + 1) * part_height)

            area = data[top:bottom, left:right]
            amount = np.count_nonzero(area < 128)

            vector.append(int(amount))

    pixels_count = sum(vector)

    if pixels_count:
        normalized = [
            round(item / pixels_count, 4)
            for item in vector
        ]
    else:
        normalized = [0.0 for _ in vector]

    marked_image = gray.convert("RGB").copy()
    painter = ImageDraw.Draw(marked_image)

    for c in range(1, COLS):
        x = int(c * part_width)
        painter.line(
            (x, 0, x, height),
            fill="red",
            width=2
        )

    for r in range(1, ROWS):
        y = int(r * part_height)
        painter.line(
            (0, y, width, y),
            fill="red",
            width=2
        )

    return vector, normalized, marked_image


def pil_to_pane(img):
    """Конвертує PIL Image у формат для Panel Pane"""
    if img is None:
        return None
    return img


def analyze_group(images, title):
    images = [img for img in images if img is not None]

    if not images:
        return (
            [],
            f"{title}: зображення відсутні.",
            ""
        )

    pictures = []
    text = []
    all_normalized = []

    for number, img in enumerate(images, 1):
        absolute, normalized, marked = get_features(img)

        pictures.append(marked)
        all_normalized.append(normalized)

        text.append(
            f"**ЗРАЗОК {number}**\n"
            f"Абсолютний вектор: `{absolute}`\n\n"
            f"Нормований вектор: `{normalized}`\n\n"
            f"{'—' * 40}"
        )

    center = np.mean(
        np.asarray(all_normalized),
        axis=0
    )

    center = [
        round(float(x), 4)
        for x in center
    ]

    info = (
        f"### {title}\n\n"
        f"**Кількість зображень:** {len(images)}\n\n"
        f"**Центр кластера:**\n`{center}`"
    )

    return pictures, "\n\n".join(text), info


def make_centers(groups):
    result = []

    for group in groups:
        group = [img for img in group if img is not None]

        if not group:
            result.append(None)
            continue

        vectors = []

        for img in group:
            _, normalized, _ = get_features(img)
            vectors.append(normalized)

        mean_vector = np.mean(
            np.asarray(vectors),
            axis=0
        )

        result.append(mean_vector.tolist())

    return result


# --- Створення інтерфейсу Panel ---

# Віджети для завантаження зображень (по 10 для кожного з 3 класів)
class_inputs = []
for i in range(3):
    class_inputs.append([pn.widgets.FileInput(accept='.png, .jpg, .jpeg, .bmp') for _ in range(10)])

# Змінна для зберігання центрів кластерів
trained_centers = None

# Виведення результатів навчання (використовуємо Markdown-панелі на всю ширину)
gallery_panes = [pn.Row(sizing_mode='stretch_width') for _ in range(3)]
vector_boxes = [pn.pane.Markdown("", sizing_mode='stretch_width', styles={'background': '#f9f9f9', 'padding': '15px', 'border-radius': '5px'}) for i in range(3)]
stats_boxes = [pn.pane.Markdown("", sizing_mode='stretch_width', styles={'background': '#f0f4f8', 'padding': '15px', 'border-radius': '5px'}) for i in range(3)]

training_message_box = pn.widgets.TextAreaInput(value="", name="Стан системи", height=120, disabled=True)

# Віджети для розпізнавання
unknown_input = pn.widgets.FileInput(accept='.png, .jpg, .jpeg, .bmp', name="Невідоме зображення")
unknown_preview = pn.pane.Image(sizing_mode='fixed', width=200, height=200)

result_absolute_box = pn.widgets.TextAreaInput(value="", name="Абсолютний вектор", height=150, disabled=True)
result_normalized_box = pn.widgets.TextAreaInput(value="", name="Нормований вектор", height=150, disabled=True)
result_image_pane = pn.pane.Image(sizing_mode='scale_both', max_height=300)
result_box = pn.widgets.TextAreaInput(value="", name="Результат", height=150, disabled=True)


def file_input_to_pil(file_input):
    if file_input.value is None:
        return None
    return Image.open(io.BytesIO(file_input.value))


def on_train_click(event):
    global trained_centers
    
    # Збираємо всі зображення з віджетів
    groups = []
    for class_idx in range(3):
        group_imgs = [file_input_to_pil(fi) for fi in class_inputs[class_idx]]
        groups.append(group_imgs)

    first = analyze_group(groups[0], "КЛАС 1")
    second = analyze_group(groups[1], "КЛАС 2")
    third = analyze_group(groups[2], "КЛАС 3")

    trained_centers = make_centers(groups)

    # Оновлюємо галереї та результати для кожного класу
    for idx, res_tuple in enumerate([first, second, third]):
        pics = res_tuple[0]
        gallery_panes[idx].objects = [pn.pane.Image(p, width=120, height=120) for p in pics]
        stats_boxes[idx].object = res_tuple[2]
        vector_boxes[idx].object = f"### Клас {idx+1} — Вектори зразків\n\n{res_tuple[1]}"

    training_message_box.value = (
        "НАВЧАННЯ ЗАВЕРШЕНО\n\n"
        "Для трьох класів визначено центри кластерів.\n"
        "Тепер можна виконати розпізнавання."
    )


train_button = pn.widgets.Button(name="Навчити систему", button_type="primary")
train_button.on_click(on_train_click)


def on_recognize_click(event):
    global trained_centers
    
    img = file_input_to_pil(unknown_input)
    if img is not None:
        unknown_preview.object = img

    if img is None:
        result_absolute_box.value = "Невідоме зображення не вибране."
        result_normalized_box.value = ""
        result_image_pane.object = None
        result_box.value = ""
        return

    if trained_centers is None:
        result_absolute_box.value = "Спочатку натисніть «Навчити систему»."
        return

    if any(item is None for item in trained_centers):
        result_absolute_box.value = "Необхідно завантажити зразки для всіх класів."
        return

    absolute, normalized, marked = get_features(img)

    results = [] 
    for center in trained_centers:
        difference = np.asarray(normalized) - np.asarray(center)
        distance = np.sqrt(np.sum(difference ** 2))
        results.append(float(distance))

    class_number = int(np.argmin(results)) + 1

    distances = (
        f"Відстань до Класу 1: {results[0]:.4f}\n"
        f"Відстань до Класу 2: {results[1]:.4f}\n"
        f"Відстань до Класу 3: {results[2]:.4f}"
    )

    answer = (
        f"РОЗПІЗНАВАННЯ\n\n"
        f"Найменша відстань: {results[class_number - 1]:.4f}\n\n"
        f"Результат: Клас {class_number}"
    )

    result_absolute_box.value = f"Абсолютний вектор:\n\n{absolute}"
    result_normalized_box.value = f"Нормований вектор:\n\n{normalized}"
    result_image_pane.object = marked
    result_box.value = distances + "\n\n" + answer


recognize_button = pn.widgets.Button(name="Розпізнати", button_type="primary")
recognize_button.on_click(on_recognize_click)


# --- Формування макету (Layout) ---

class_tabs = []
for c_idx in range(3):
    inputs_grid = pn.GridSpec(sizing_mode='stretch_width', max_height=250)
    for sample_idx, fi in enumerate(class_inputs[c_idx]):
        row_pos = sample_idx // 5
        col_pos = sample_idx % 5
        inputs_grid[row_pos, col_pos] = pn.Column(f"Зразок {c_idx+1}.{sample_idx+1}", fi)

    tab_content = pn.Column(
        pn.pane.Markdown(f"### Завантаження зразків для Класу {c_idx+1}"),
        inputs_grid,
        pn.Spacer(height=10),
        stats_boxes[c_idx],
        vector_boxes[c_idx],
        pn.pane.Markdown("#### Результати обробки класу (сітка 5×5):"),
        gallery_panes[c_idx],
        sizing_mode='stretch_width'
    )
    class_tabs.append((f"Клас {c_idx+1}", tab_content))

tabs_widget = pn.Tabs(*class_tabs)

training_section = pn.Column(
    tabs_widget,
    pn.Spacer(height=15),
    train_button,
    training_message_box,
    sizing_mode='stretch_width'
)

recognition_section = pn.Column(
    pn.pane.Markdown("## Розпізнавання невідомого образу"),
    pn.Row(unknown_input, unknown_preview),
    recognize_button,
    pn.Row(result_absolute_box, result_normalized_box),
    pn.Row(result_image_pane, result_box),
    sizing_mode='stretch_width'
)

app = pn.Column(
    pn.pane.Markdown("# Система розпізнавання образів\nДля навчання використовуються три класи, по 10 зображень у кожному."),
    training_section,
    pn.layout.Divider(),  
    recognition_section,
    sizing_mode='stretch_width'
)

if __name__ == "__main__":
    pn.serve(app, port=5006, show=True)
else:
    app.servable()
