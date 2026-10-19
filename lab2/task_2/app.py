import io

import numpy as np
import panel as pn
from PIL import Image, ImageDraw

pn.extension()

GRID_ROWS = 5
GRID_COLS = 5
N_FEATURES = GRID_ROWS * GRID_COLS
CLASS_NAMES = ("Клас 1", "Клас 2", "Клас 3")
MAX_HOPFIELD_STEPS = 100
MAX_HAMMING_STEPS = 100


def load_image(data):
    """Безпечно декодує файл FileInput і повертає RGB-зображення."""
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (OSError, ValueError):
        return None


def extract_features(image):
    """Рахує кількість темних пікселів у кожній комірці сітки 5×5."""
    if image is None:
        return None, None, None

    gray = image.convert("L")
    pixels = np.asarray(gray)
    height, width = pixels.shape
    absolute = np.zeros(N_FEATURES, dtype=float)

    # Межі через linspace рівномірно охоплюють і зображення, і залишкові пікселі.
    y_edges = np.linspace(0, height, GRID_ROWS + 1, dtype=int)
    x_edges = np.linspace(0, width, GRID_COLS + 1, dtype=int)
    index = 0
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cell = pixels[y_edges[row]:y_edges[row + 1], x_edges[col]:x_edges[col + 1]]
            absolute[index] = np.count_nonzero(cell < 128)
            index += 1

    # L2-нормування: масштаб вектора стає одиничним, напрямок ознак зберігається.
    norm = float(np.linalg.norm(absolute))
    normalized = absolute / norm if norm > 0 else np.zeros_like(absolute)

    preview = gray.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    for x in x_edges[1:-1]:
        draw.line((int(x), 0, int(x), height), fill=(230, 50, 50), width=2)
    for y in y_edges[1:-1]:
        draw.line((0, int(y), width, int(y)), fill=(230, 50, 50), width=2)
    return absolute, normalized, preview


def bipolarize(vector):
    """Перетворює ознаки в {-1,+1}: наявність темних пікселів → +1."""
    return np.where(np.asarray(vector) > 0, 1.0, -1.0)


def format_vector(vector, precision=3):
    if vector is None:
        return "—"
    return "[" + ", ".join(f"{x:.{precision}f}" for x in vector) + "]"


def make_hopfield_weights(patterns):
    """Hebb: W = (1/N) Σ xμ xμᵀ, діагональ якої дорівнює нулю."""
    matrix = np.asarray(patterns, dtype=float)
    weights = matrix.T @ matrix / matrix.shape[1]
    np.fill_diagonal(weights, 0.0)
    return weights


def run_hopfield(query, weights):
    """Асинхронно оновлює нейрони; повертає стан, трасу і факт стабілізації."""
    state = np.asarray(query, dtype=float).copy()
    trace = [state.copy()]
    seen = {tuple(state.astype(int))}
    for _ in range(MAX_HOPFIELD_STEPS):
        changed = False
        for i in range(state.size):
            activation = float(weights[i] @ state)
            new_value = 1.0 if activation > 0 else (-1.0 if activation < 0 else state[i])
            if new_value != state[i]:
                state[i] = new_value
                changed = True
        key = tuple(state.astype(int))
        trace.append(state.copy())
        if not changed:
            return state, trace, True, "Стан стабілізувався."
        if key in seen:
            return state, trace, False, "Виявлено цикл станів; мережа не стабілізувалася."
        seen.add(key)
    return state, trace, False, f"Досягнуто ліміт {MAX_HOPFIELD_STEPS} проходів."


def hopfield_energy(state, weights):
    return float(-0.5 * state @ weights @ state)


def hamming_first_layer(patterns):
    """Ваги першого шару: wᵢⱼ = xᵢⱼ/2; зсув для кожного класу — N/2."""
    return np.asarray(patterns, dtype=float).T / 2.0, N_FEATURES / 2.0


def run_hamming(query, patterns):
    """Hamming net: шар збігів і конкурентне латеральне гальмування."""
    weights, bias = hamming_first_layer(patterns)
    first_layer = weights.T @ query + bias
    outputs = first_layer.copy()
    trace = [outputs.copy()]
    # ε обрано так, щоб за скінченну кількість кроків пригнічувалися всі,
    # крім переможця; argmax дає детермінований вибір при нічиїй.
    epsilon = 1.0 / (2.0 * len(CLASS_NAMES))
    winner = int(np.argmax(outputs))
    active = np.ones(len(CLASS_NAMES), dtype=bool)
    for _ in range(MAX_HAMMING_STEPS):
        if np.count_nonzero(active) <= 1:
            break
        next_outputs = outputs.copy()
        for i in range(len(outputs)):
            if active[i]:
                inhibition = epsilon * sum(outputs[j] for j in range(len(outputs)) if j != i and active[j])
                next_outputs[i] = max(0.0, outputs[i] - inhibition)
        outputs = next_outputs
        trace.append(outputs.copy())
        survivors = np.flatnonzero(outputs > 0)
        if survivors.size == 1:
            winner = int(survivors[0])
            break
        if survivors.size == 0:
            winner = int(np.argmax(first_layer))
            break
        active = outputs > 0
        winner = int(np.argmax(outputs))
    return winner, first_layer, outputs, trace, weights, bias


def matrix_text(matrix, row_labels=None, precision=3):
    matrix = np.asarray(matrix)
    labels = row_labels or [f"{i + 1:02d}" for i in range(matrix.shape[0])]
    lines = ["      " + " ".join(f"{j + 1:>7}" for j in range(matrix.shape[1]))]
    for label, row in zip(labels, matrix):
        lines.append(f"{label:>5} " + " ".join(f"{value:7.{precision}f}" for value in row))
    return "\n".join(lines)


def make_app():
    state = {"images": [None] * 3, "features": [None] * 3, "query": None}
    model = pn.widgets.RadioButtonGroup(
        name="Модель мережі", options=["Хопфілда", "Хеммінга"], value="Хопфілда",
        button_type="primary",
    )
    class_uploads = [
        pn.widgets.FileInput(name=f"Еталон {name} (PNG/JPG)", accept=".png,.jpg,.jpeg")
        for name in CLASS_NAMES
    ]
    unknown_upload = pn.widgets.FileInput(name="Невідомий образ (PNG/JPG)", accept=".png,.jpg,.jpeg")
    # Фіксована висота + адаптивна ширина: scale_both тут спричиняв
    # конфлікт розмірів, через який наступні картки заходили на попередні.
    class_images = [pn.pane.Image(height=180, sizing_mode="scale_both") for _ in CLASS_NAMES]
    class_grids = [pn.pane.Image(height=180, sizing_mode="scale_both") for _ in CLASS_NAMES]
    class_images = [pn.pane.Image(height=180, sizing_mode="stretch_width") for _ in CLASS_NAMES]
    class_grids = [pn.pane.Image(height=180, sizing_mode="stretch_width") for _ in CLASS_NAMES]
    class_features = [pn.widgets.TextAreaInput(name=f"Ознаки {name}", height=110, sizing_mode="stretch_width") for name in CLASS_NAMES]
    query_image = pn.pane.Image(height=240, sizing_mode="scale_both")
    query_grid = pn.pane.Image(height=240, sizing_mode="scale_both")
    query_image = pn.pane.Image(height=240, sizing_mode="stretch_width")
    query_grid = pn.pane.Image(height=240, sizing_mode="stretch_width")
    query_features = pn.widgets.TextAreaInput(name="Ознаки невідомого образу", height=140, sizing_mode="stretch_width")
    matrix_output = pn.widgets.TextAreaInput(name="Матриця мережі", height=430, sizing_mode="stretch_width")
    result_output = pn.widgets.TextAreaInput(name="Класифікація та аналіз", height=300, sizing_mode="stretch_width")
    classify_button = pn.widgets.Button(name="Побудувати мережу та класифікувати", button_type="success", width=300)
    reset_button = pn.widgets.Button(name="Очистити все", button_type="warning")
    status = pn.pane.Markdown("Завантажте три еталонні зображення та невідомий образ.")

    def update_reference(index, event):
        image = load_image(event.new)
        state["images"][index] = image
        absolute, normalized, grid = extract_features(image)
        state["features"][index] = None if absolute is None else (absolute, normalized)
        class_images[index].object = image
        class_grids[index].object = grid
        if image is None:
            class_features[index].value = "Некоректний файл. Завантажте PNG або JPG."
        else:
            class_features[index].value = (
                f"Абсолютний вектор ({N_FEATURES} ознак):\n{format_vector(absolute, 0)}\n\n"
                f"L2-нормований вектор:\n{format_vector(normalized, 4)}\n"
                f"Норма до нормування: {np.linalg.norm(absolute):.3f}"
            )

    for idx, uploader in enumerate(class_uploads):
        uploader.param.watch(lambda event, i=idx: update_reference(i, event), "value")

    def update_query(event):
        image = load_image(event.new)
        state["query"] = image
        absolute, normalized, grid = extract_features(image)
        query_image.object = image
        query_grid.object = grid
        if image is None:
            query_features.value = "Завантажте коректне зображення PNG або JPG."
        else:
            query_features.value = (
                f"Абсолютний вектор:\n{format_vector(absolute, 0)}\n\n"
                f"L2-нормований вектор:\n{format_vector(normalized, 4)}"
            )

    unknown_upload.param.watch(update_query, "value")

    def classify(event=None):
        if any(x is None for x in state["features"]):
            status.object = "**Не вистачає даних:** завантажте еталонне зображення для кожного з трьох класів."
            return
        if state["query"] is None:
            status.object = "**Не вистачає даних:** завантажте невідоме зображення."
            return
        ref_absolute = np.stack([pair[0] for pair in state["features"]])
        ref_normalized = np.stack([pair[1] for pair in state["features"]])
        query_absolute, query_normalized, _ = extract_features(state["query"])
        if np.any(np.linalg.norm(ref_absolute, axis=1) == 0) or np.linalg.norm(query_absolute) == 0:
            status.object = "**Нульовий вектор ознак.** Зображення без темних пікселів не містить ознак для класифікації."
            return

        if model.value == "Хопфілда":
            patterns = np.stack([bipolarize(v) for v in ref_absolute])
            query = bipolarize(query_absolute)
            weights = make_hopfield_weights(patterns)
            recalled, trace, stable, convergence = run_hopfield(query, weights)
            distances = np.count_nonzero(patterns != recalled, axis=1)
            winner = int(np.argmin(distances))
            energies = [hopfield_energy(v, weights) for v in trace]
            matrix_output.value = (
                "Мережа Хопфілда — матриця ваг W (25×25)\n"
                "Біполяризація еталонів: ознака > 0 → +1, інакше → −1.\n"
                "Правило: W = (1/25) Σ x·xᵀ; діагональ W дорівнює 0.\n\n"
                + matrix_text(weights)
                + "\n\nЕталонні біполярні вектори:\n"
                + "\n".join(f"{name}: {format_vector(patterns[i], 0)}" for i, name in enumerate(CLASS_NAMES))
            )
            result_output.value = (
                f"Результат: {CLASS_NAMES[winner]}\n"
                f"Збіжність: {convergence}\n"
                f"Кроків мережі: {len(trace) - 1}\n"
                f"Енергія E = −½·sᵀWs: {energies[0]:.3f} → {energies[-1]:.3f}\n"
                f"Енергія не зростала: {'так' if all(b <= a + 1e-9 for a, b in zip(energies, energies[1:])) else 'ні'}\n"
                f"Відстань Хеммінга від згаданого стану до еталонів:\n"
                + "\n".join(f"  {name}: {distances[i]} з 25" for i, name in enumerate(CLASS_NAMES))
                + f"\n\nСтабільність мережі: {'стабільний атрактор' if stable else 'можливий цикл або ліміт ітерацій'}.\n"
                "Клас визначено за еталоном, найближчим до стану після рекурентного оновлення."
            )
        else:
            # Для Хеммінга вхід і еталони — одиничні L2-вектори ознак.
            patterns = ref_normalized
            query = query_normalized
            winner, first, final, trace, weights, bias = run_hamming(query, patterns)
            similarity = patterns @ query
            matrix_output.value = (
                "Мережа Хеммінга — ваги першого шару W₁ (25×3)\n"
                "Еталонні вектори нормовано за L2; ваги wᵢⱼ = xᵢⱼ/2, зсув θ = N/2 = 12.5.\n"
                "Перший шар обчислює N/2 + (xⱼᵀ·x)/2; далі працює конкурентне гальмування.\n\n"
                + matrix_text(weights, [f"f{i + 1:02d}" for i in range(N_FEATURES)])
                + "\n\nВаги другого шару (латеральне гальмування):\n"
                + matrix_text(-np.ones((3, 3)) + np.eye(3), CLASS_NAMES)
                + "\n\nЗміщення першого шару: " + format_vector(np.full(3, bias), 1)
            )
            result_output.value = (
                f"Результат: {CLASS_NAMES[winner]}\n"
                f"Виходи першого шару (N/2 + збіг): {format_vector(first, 4)}\n"
                f"Схожість нормованих ознак (cosine): {format_vector(similarity, 4)}\n"
                f"Виходи конкурентного шару: {format_vector(final, 4)}\n"
                f"Ітерацій конкурентного шару: {len(trace) - 1}\n"
                "Аналіз: клас-переможець має найбільший скалярний добуток із нормованим еталоном; інші нейрони пригнічуються.\n"
                "За однакового максимального виходу обирається перший клас у списку."
            )
        status.object = f"**Готово.** Класифікацію виконано мережею {model.value}."

    def reset(event=None):
        for uploader in class_uploads:
            uploader.value = None
        unknown_upload.value = None
        for i in range(3):
            state["images"][i] = None
            state["features"][i] = None
            class_images[i].object = None
            class_grids[i].object = None
            class_features[i].value = ""
        state["query"] = None
        query_image.object = None
        query_grid.object = None
        query_features.value = ""
        matrix_output.value = ""
        result_output.value = ""
        status.object = "Дані очищено. Завантажте три еталонні зображення та невідомий образ."

    classify_button.on_click(classify)
    reset_button.on_click(reset)

    reference_cards = []
    for i, name in enumerate(CLASS_NAMES):
        reference_cards.append(pn.Card(
            class_uploads[i],
            pn.Row(
                pn.Column("Зображення", class_images[i], sizing_mode="stretch_width"),
                pn.Column("Сітка ознак 5×5", class_grids[i], sizing_mode="stretch_width"),
            ),
            class_features[i], title=name, collapsed=False,
        ))

    # Базовий Bokeh layout без шаблонної оболонки. Так менше залежності від
    # завантаження додаткових стилів та клієнтського коду шаблону Panel.
    return pn.Column(
        pn.pane.Markdown(
            "# Система розпізнавання трьох класів\n"
            "Кожне зображення ділиться на сітку 5×5. Ознака комірки — кількість пікселів темніших за 128. "
            "Абсолютні та L2-нормовані вектори показано для кожного зображення."
        ),
        pn.Card(*reference_cards, title="1. Еталонні зображення трьох класів", collapsed=False),
        pn.Card(
            unknown_upload,
            pn.Row(
                pn.Column("Невідоме зображення", query_image, sizing_mode="stretch_width"),
                pn.Column("Сітка ознак 5×5", query_grid, sizing_mode="stretch_width"),
            ),
            query_features,
            title="2. Невідомий образ та його ознаки", collapsed=False,
        ),
        pn.Card(
            pn.Row(model, classify_button, reset_button), status,
            pn.Row(matrix_output, result_output),
            title="3. Побудова мережі, класифікація та аналіз", collapsed=False,
        ),
        sizing_mode="stretch_width",
    )


app = make_app()
app.servable()