
import io
import random

import numpy as np
import panel as pn
from PIL import Image, ImageDraw

pn.extension()

# -------------------- Параметри системи --------------------
GRID_ROWS = 5
GRID_COLS = 5
R_DEFAULT = 0.5
MAX_EPOCHS = 100
CLASS_1 = -1
CLASS_2 = 1


# -------------------- Обробка зображення --------------------
def load_pil_image(file_bytes):
    """Перетворює байти завантаженого графічного файлу на PIL.Image."""
    if not file_bytes:
        return None
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return None


def calculate_feature_vector(image):
    """Повертає абсолютний вектор, нормований вектор і зображення із сіткою 5×5."""
    if image is None:
        return None, None, None

    gray_image = image.convert("L")
    width, height = gray_image.size
    pixels = np.asarray(gray_image)

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
            absolute_vector.append(int(np.sum(cell < 128)))

    total_black_pixels = sum(absolute_vector)
    if total_black_pixels:
        normalized_vector = [value / total_black_pixels for value in absolute_vector]
    else:
        normalized_vector = [0.0] * (GRID_ROWS * GRID_COLS)

    grid_image = gray_image.convert("RGB").copy()
    draw = ImageDraw.Draw(grid_image)
    for col in range(1, GRID_COLS):
        x = int(col * cell_width)
        draw.line((x, 0, x, height), fill="red", width=2)
    for row in range(1, GRID_ROWS):
        y = int(row * cell_height)
        draw.line((0, y, width, y), fill="red", width=2)

    return absolute_vector, normalized_vector, grid_image


# -------------------- Перцептрон Розенблата --------------------
def create_perceptron():
    weights = [random.uniform(0, 1) for _ in range(GRID_ROWS * GRID_COLS)]
    bias = random.uniform(0, 1)
    return weights, bias


def predict(normalized_vector, weights, bias):
    x = np.asarray(normalized_vector, dtype=float)
    w = np.asarray(weights, dtype=float)
    weighted_sum = float(np.dot(x, w) + bias)
    output = CLASS_2 if weighted_sum >= 0 else CLASS_1
    return weighted_sum, output


def get_class_name(output):
    return "Клас 1" if output == CLASS_1 else "Клас 2"


def update_perceptron(normalized_vector, desired_output, weights, bias, learning_rate):
    weighted_sum, output = predict(normalized_vector, weights, bias)
    error = desired_output - output
    if output != desired_output:
        x = np.asarray(normalized_vector, dtype=float)
        w = np.asarray(weights, dtype=float)
        w = w + learning_rate * error * x
        bias = bias + learning_rate * error
        weights = w.tolist()
    return weights, bias, weighted_sum, output, error


def format_weights(weights):
    if weights is None:
        return "Перцептрон ще не створено."
    return "\n".join(f"w{i} = {weight:.6f}" for i, weight in enumerate(weights, 1))


def format_vectors(absolute_vector, normalized_vector):
    if absolute_vector is None or normalized_vector is None:
        return "Завантажте зображення, щоб обчислити вектори ознак."
    normalized_text = [round(value, 6) for value in normalized_vector]
    return (
        f"Абсолютний вектор ознак ({len(absolute_vector)} значень):\n"
        f"{absolute_vector}\n\n"
        "----------------------------------------\n\n"
        f"Нормований вектор ознак ({len(normalized_vector)} значень):\n"
        f"{normalized_text}\n\n"
        f"Сума нормованих ознак: {sum(normalized_vector):.6f}"
    )


# -------------------- Інтерфейс Panel --------------------
def make_app():
    # Стан конкретного екземпляра застосунку
    state = {
        "weights": None,
        "bias": None,
        "step": 0,
        "training_samples": [],
        "current_image": None,
    }

    learning_rate = pn.widgets.FloatInput(
        name="Швидкість навчання R", value=R_DEFAULT,
        start=0.01, end=1.0, step=0.05, width=250
    )
    initialize_button = pn.widgets.Button(
        name="Створити перцептрон", button_type="primary", width=220
    )
    reset_button = pn.widgets.Button(name="Скинути систему", button_type="warning", width=180)

    initial_status = pn.pane.Markdown("**Стан системи:** перцептрон ще не створено.")
    bias_output = pn.widgets.StaticText(name="Поточне зміщення b", value="—", width=220)
    step_output = pn.widgets.StaticText(name="Крок навчання", value="0", width=180)
    weights_output = pn.widgets.TextAreaInput(
        name="Поточні ваги перцептрона", value="", height=230, sizing_mode="stretch_width"
    )

    upload = pn.widgets.FileInput(name="Завантажити образ (PNG/JPG/JPEG)", accept=".png,.jpg,.jpeg")
    correct_class = pn.widgets.RadioButtonGroup(
        name="Правильний клас (навчання з учителем)",
        options=["Клас 1", "Клас 2"], value="Клас 1", button_type="success"
    )
    recognize_button = pn.widgets.Button(name="Розпізнати образ", button_type="primary")
    train_button = pn.widgets.Button(name="Підтвердити / скоригувати", button_type="success")
    image_preview = pn.pane.Image(height=300, sizing_mode="scale_both")
    grid_preview = pn.pane.Image(height=300, sizing_mode="scale_both")
    vectors_output = pn.widgets.TextAreaInput(
        name="Абсолютний і нормований вектори ознак", value="", height=220,
        sizing_mode="stretch_width"
    )
    recognition_output = pn.widgets.TextAreaInput(
        name="Результат розпізнавання", value="", height=140, sizing_mode="stretch_width"
    )
    training_output = pn.widgets.TextAreaInput(
        name="Результат навчання та корекції", value="", height=220, sizing_mode="stretch_width"
    )
    convergence_button = pn.widgets.Button(
        name="Перевірити збіжність", button_type="primary", width=220
    )
    convergence_output = pn.widgets.TextAreaInput(
        name="Результат оцінювання збіжності", value="", height=220, sizing_mode="stretch_width"
    )

    def show_current_image(event=None):
        image = load_pil_image(upload.value)
        state["current_image"] = image
        if image is None:
            image_preview.object = None
            grid_preview.object = None
            vectors_output.value = "Завантажте коректний графічний файл PNG або JPG."
            return
        image_preview.object = image
        absolute, normalized, grid = calculate_feature_vector(image)
        vectors_output.value = format_vectors(absolute, normalized)
        grid_preview.object = grid

    upload.param.watch(show_current_image, "value")

    def initialize(event=None):
        rate = learning_rate.value or R_DEFAULT
        if rate <= 0:
            rate = R_DEFAULT
        weights, bias = create_perceptron()
        state.update(weights=weights, bias=bias, step=0, training_samples=[])
        weights_output.value = format_weights(weights)
        bias_output.value = f"{bias:.6f}"
        step_output.value = "0"
        initial_status.object = (
            "### Перцептрон створено\n"
            f"- Кількість входів і ваг: **{len(weights)}**\n"
            f"- Швидкість навчання R: **{rate}**\n"
            f"- Початкове зміщення b: **{bias:.6f}**\n"
            "- Навчальних прикладів поки немає."
        )
        recognition_output.value = ""
        training_output.value = ""
        convergence_output.value = ""

    def recognize(event=None):
        image = state["current_image"]
        if image is None:
            recognition_output.value = "Завантажте зображення."
            return
        if state["weights"] is None:
            recognition_output.value = "Спочатку створіть перцептрон."
            return
        absolute, normalized, grid = calculate_feature_vector(image)
        weighted_sum, output = predict(normalized, state["weights"], state["bias"])
        vectors_output.value = format_vectors(absolute, normalized)
        grid_preview.object = grid
        recognition_output.value = (
            f"Зважена сума S = {weighted_sum:.6f}\n"
            f"Зміщення b = {state['bias']:.6f}\n"
            f"Вихід перцептрона y = {output}\n"
            f"Розпізнаний клас: {get_class_name(output)}"
        )

    def train(event=None):
        image = state["current_image"]
        weights, bias = state["weights"], state["bias"]
        if image is None:
            training_output.value = "Спочатку завантажте зображення."
            return
        if weights is None or bias is None:
            training_output.value = "Спочатку створіть перцептрон."
            return

        rate = learning_rate.value or R_DEFAULT
        if rate <= 0:
            rate = R_DEFAULT

        absolute, normalized, grid = calculate_feature_vector(image)
        desired = CLASS_1 if correct_class.value == "Клас 1" else CLASS_2
        weighted_sum, output = predict(normalized, weights, bias)
        error = desired - output

        # Зберігаємо навчальний зразок, не додаючи точний дублікат з тією самою міткою.
        already_saved = any(
            sample["x"] == normalized and sample["target"] == desired
            for sample in state["training_samples"]
        )
        if not already_saved:
            state["training_samples"].append({
                "x": normalized.copy(), "target": desired, "label": correct_class.value
            })

        state["step"] += 1
        if output == desired:
            result = (
                "КЛАСИФІКАЦІЯ ПРАВИЛЬНА\n"
                f"Розпізнаний клас: {get_class_name(output)}\n"
                f"Правильний клас: {correct_class.value}\n"
                f"Вихід y = {output}; бажаний вихід d = {desired}\n"
                "Помилка класифікації відсутня. Ваги та зміщення не змінюються."
            )
        else:
            weights, bias, _, _, _ = update_perceptron(
                normalized, desired, weights, bias, rate
            )
            state["weights"], state["bias"] = weights, bias
            new_sum, new_output = predict(normalized, weights, bias)
            result = (
                "КЛАСИФІКАЦІЯ БУЛА НЕПРАВИЛЬНОЮ\n"
                f"Попередній клас: {get_class_name(output)}; правильний клас: {correct_class.value}\n"
                f"Помилка d - y = {error}\n"
                "Ваги та зміщення скориговано.\n"
                f"Нова зважена сума S = {new_sum:.6f}\n"
                f"Новий вихід y = {new_output}\n"
                f"Клас після корекції: {get_class_name(new_output)}\n"
                f"Результат після одного оновлення: "
                f"{'правильний' if new_output == desired else 'ще неправильний'}"
            )

        # Якщо класифікація була правильною, параметри залишаються без змін.
        state["weights"], state["bias"] = weights, bias
        weights_output.value = format_weights(weights)
        bias_output.value = f"{bias:.6f}"
        step_output.value = str(state["step"])
        vectors_output.value = format_vectors(absolute, normalized)
        grid_preview.object = grid
        training_output.value = (
            result + f"\n\nКрок навчання: {state['step']}"
            + f"\nШвидкість навчання R = {rate}"
            + f"\nЗбережено навчальних прикладів: {len(state['training_samples'])}"
            + f"\nЗміщення b = {bias:.6f}"
        )
        recognize()

    def check_convergence(event=None):
        weights, bias = state["weights"], state["bias"]
        samples = state["training_samples"]
        if weights is None or bias is None:
            convergence_output.value = "Спочатку створіть перцептрон."
            return
        if not samples:
            convergence_output.value = "Навчальний набір порожній. Додайте приклади обох класів."
            return
        rate = learning_rate.value or R_DEFAULT
        if rate <= 0:
            rate = R_DEFAULT

        initial_weights = weights.copy()
        initial_bias = bias
        epochs_used = 0
        mistakes = len(samples)

        for epoch in range(1, MAX_EPOCHS + 1):
            for sample in samples:
                weights, bias, _, _, _ = update_perceptron(
                    sample["x"], sample["target"], weights, bias, rate
                )
                state["step"] += 1
            mistakes = sum(
                predict(sample["x"], weights, bias)[1] != sample["target"]
                for sample in samples
            )
            epochs_used = epoch
            if mistakes == 0:
                break

        correct = len(samples) - mistakes
        accuracy = correct / len(samples) * 100
        converged = mistakes == 0
        conclusion = (
            "ЗБІЖНІСТЬ ДОСЯГНУТО: усі навчальні приклади класифікуються правильно."
            if converged else
            f"ЗБІЖНІСТЬ НЕ ДОСЯГНУТО за {MAX_EPOCHS} епох. Перевірте приклади та їхні мітки."
        )
        changed = weights != initial_weights or bias != initial_bias

        state["weights"], state["bias"] = weights, bias
        weights_output.value = format_weights(weights)
        bias_output.value = f"{bias:.6f}"
        step_output.value = str(state["step"])
        initial_status.object = (
            f"Перевірено {len(samples)} навчальних прикладів. "
            f"Виконано епох: {epochs_used}. Залишилося помилок: {mistakes}."
        )
        convergence_output.value = (
            f"Кількість навчальних прикладів: {len(samples)}\n"
            f"Кількість виконаних епох: {epochs_used}\n"
            f"Загальний лічильник кроків: {state['step']}\n"
            f"Правильних класифікацій: {correct}\n"
            f"Помилок після останньої епохи: {mistakes}\n"
            f"Точність на навчальному наборі: {accuracy:.2f}%\n"
            f"Швидкість навчання R: {rate}\n\n{conclusion}\n\n"
            f"Початкове зміщення b: {initial_bias:.6f}\n"
            f"Поточне зміщення b: {bias:.6f}\n"
            f"Параметри змінилися: {'так' if changed else 'ні'}"
        )

    def reset(event=None):
        state.update(weights=None, bias=None, step=0, training_samples=[], current_image=None)
        upload.value = None
        image_preview.object = None
        grid_preview.object = None
        weights_output.value = ""
        bias_output.value = "—"
        step_output.value = "0"
        vectors_output.value = ""
        recognition_output.value = ""
        training_output.value = ""
        convergence_output.value = ""
        initial_status.object = "Систему скинуто. Створіть новий перцептрон."

    initialize_button.on_click(initialize)
    reset_button.on_click(reset)
    recognize_button.on_click(recognize)
    train_button.on_click(train)
    convergence_button.on_click(check_convergence)

    return pn.template.FastListTemplate(
        title="Перцептрон Розенблата",
        main=[
            pn.pane.Markdown(
                "# Система розпізнавання перцептронного типу\n"
                "Двокласова класифікація зображень. Кожне зображення ділиться на сітку 5×5; "
                "для 25 комірок обчислюються абсолютні та нормовані числові ознаки."
            ),
            pn.Card(
                pn.Row(learning_rate, initialize_button, reset_button),
                initial_status,
                pn.Row(bias_output, step_output),
                weights_output,
                title="1. Початкові параметри перцептрона", collapsed=False
            ),
            pn.Card(
                upload,
                pn.Row(
                    pn.Column("### Завантажений образ", image_preview, sizing_mode="stretch_width"),
                    pn.Column("### Сітка 5×5", grid_preview, sizing_mode="stretch_width"),
                ),
                vectors_output,
                correct_class,
                pn.Row(recognize_button, train_button),
                recognition_output,
                training_output,
                title="2. Послідовне введення, розпізнавання та навчання", collapsed=False
            ),
            pn.Card(
                pn.pane.Markdown(
                    "Завантажуйте навчальні зразки **по одному**. Для кожного виберіть правильний клас "
                    "і натисніть «Підтвердити / скоригувати». Додайте приклади обох класів, "
                    "перш ніж оцінювати збіжність."
                ),
                convergence_button,
                convergence_output,
                title="3. Оцінювання збіжності", collapsed=False
            ),
        ],
        accent_base_color="#2864a5",
        header_background="#163b63",
    )


app = make_app()

if __name__ == "__main__":
    pn.serve(app, show=True, title="Перцептрон Розенблата")
