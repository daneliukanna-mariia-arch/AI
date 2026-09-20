import os
from PIL import Image, ImageDraw

# Створюємо папку для зразків, якщо її немає
os.makedirs("dataset", exist_ok=True)

# Генеруємо по 10 зразків для кожного класу з невеликими варіаціями
for i in range(1, 11):
  # --- Клас 1: Кола з різним розміром/зсувом ---
  img1 = Image.new("RGB", (100, 100), "white")
  draw1 = ImageDraw.Draw(img1)
  offset = i * 1.5
  draw1.ellipse(
      [30 + offset, 30 + offset, 70 - offset, 70 - offset], fill="black"
  )
  img1.save(f"dataset/class_1_{i}.bmp")

  # --- Клас 2: Квадрати ---
  img2 = Image.new("RGB", (100, 100), "white")
  draw2 = ImageDraw.Draw(img2)
  draw2.rectangle(
      [25 + offset, 25 + offset, 75 - offset, 75 - offset], fill="black"
  )
  img2.save(f"dataset/class_2_{i}.bmp")

  # --- Клас 3: Трикутники ---
  img3 = Image.new("RGB", (100, 100), "white")
  draw3 = ImageDraw.Draw(img3)
  draw3.polygon(
      [
          (50, 20 + offset),
          (20 + offset, 80 - offset),
          (80 - offset, 80 - offset),
      ],
      fill="black",
  )
  img3.save(f"dataset/class_3_{i}.bmp")

# Невідомий образ для перевірки
img_unk = Image.new("RGB", (100, 100), "white")
draw_unk = ImageDraw.Draw(img_unk)
draw_unk.ellipse([35, 35, 65, 65], fill="black")
img_unk.save("dataset/unknown.bmp")

print("Серії з по 10 зразків успішно створені в папці 'dataset'! 🎉")
