import io
from PIL import Image
import panel as pn

# Ініціалізація Panel
pn.extension()

# Заголовок та опис
title = pn.pane.Markdown("# Lab 1 - Image Uploader (Panel)")
info_pane = pn.pane.Alert("Please upload an image file.", alert_type="info")

# Віджет завантаження файлу
file_input = pn.widgets.FileInput(accept='.jpg,.jpeg,.png,.webp', name='Pick a file')

# Обгортка для виведення зображення або повідомлення
@pn.depends(file_input.param.value)
def update_image(value):
    if value is not None:
        try:
            image_bytes = file_input.value
            image = Image.open(io.BytesIO(image_bytes))
            # Повертаємо зображення через pn.pane.Image
            return pn.Column(
                pn.pane.Alert("Image successfully uploaded!", alert_type="success"),
                pn.pane.Image(image, width=400, caption='Uploaded Image')
            )
        except Exception as e:
            return pn.pane.Alert(f"Could not open image: {e}", alert_type="danger")
    else:
        return info_pane

# Компонування макету
layout = pn.Column(
    title,
    "A simple Panel app that allows you to upload an image (`jpg`, `jpeg`, `png`, `webp`) and display it.",
    file_input,
    update_image
)

# Робимо додаток доступним для запуску через сервер
layout.servable()