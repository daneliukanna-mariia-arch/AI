
from PIL import Image, ImageDraw

# 1. Еталон 1 (наприклад, круг)
img1 = Image.new("RGB", (100, 100), "white")
draw1 = ImageDraw.Draw(img1)
draw1.ellipse([30, 30, 70, 70], fill="black")
img1.save("class_1.bmp")

# 2. Еталон 2 (наприклад, квадрат)
img2 = Image.new("RGB", (100, 100), "white")
draw2 = ImageDraw.Draw(img2)
draw2.rectangle([25, 25, 75, 75], fill="black")
img2.save("class_2.bmp")

# 3. Еталон 3 (наприклад, трикутник)
img3 = Image.new("RGB", (100, 100), "white")
draw3 = ImageDraw.Draw(img3)
draw3.polygon([(50, 20), (20, 80), (80, 80)], fill="black")
img3.save("class_3.bmp")

# 4. Невідомий образ (зробимо його схожим на Еталон 1, наприклад, трохи зміщений або менший круг)
img_unk = Image.new("RGB", (100, 100), "white")
draw_unk = ImageDraw.Draw(img_unk)
draw_unk.ellipse([35, 35, 65, 65], fill="black")
img_unk.save("unknown.bmp")

print(
    "Готово! Файли class_1.bmp, class_2.bmp, class_3.bmp та unknown.bmp створено."
)
