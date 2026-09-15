
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class FeatureExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторна 1 - Завдання 1: Вектори ознак")
        self.root.geometry("700x500")

        self.grid_rows = 5
        self.grid_cols = 4

        self.btn_load = tk.Button(root, text="Завантажити зображення (.bmp)", command=self.load_image)
        self.btn_load.pack(pady=10)

        self.lbl_img = tk.Label(root)
        self.lbl_img.pack()

        self.text_output = tk.Text(root, height=12, width=80)
        self.text_output.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")])
        if not file_path:
            return
        
        img = Image.open(file_path).convert('L')
        
        display_img = img.resize((200, 250))
        self.photo = ImageTk.PhotoImage(display_img)
        self.lbl_img.config(image=self.photo)

        absolute_vector, normalized_vector = self.extract_features(img)

        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, f"Абсолютний вектор ознак (розмір {len(absolute_vector)}):\n{absolute_vector}\n\n")
        self.text_output.insert(tk.END, f"Нормований вектор ознак:\n{normalized_vector}")

    def extract_features(self, img):
        img_np = np.array(img)
        h, w = img_np.shape
        
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols
        
        absolute_vector = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                
                cell = img_np[y1:y2, x1:x2]
                dark_pixels = np.sum(cell < 128)  # Підрахунок темних пікселів
                absolute_vector.append(int(dark_pixels))
                
        abs_arr = np.array(absolute_vector, dtype=float)
        norm_sum = np.sum(abs_arr)
        normalized_vector = (abs_arr / norm_sum).tolist() if norm_sum > 0 else abs_arr.tolist()
            
        return absolute_vector, normalized_vector

if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureExtractorApp(root)
    root.mainloop()

