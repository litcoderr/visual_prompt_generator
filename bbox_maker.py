import os
import numpy as np
import tkinter as tk

from pathlib import Path
from tkinter import filedialog
from PIL import Image, ImageTk

class BoundingBoxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bounding Box Drawer")

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.img_root = Path(filedialog.askdirectory(title="Select Image Directory"))
        self.imgs = [f for f in os.listdir(self.img_root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.imgs.sort()

        self.idx = 0
        self.init_image()

        # Bind mouse events for drawing
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Right>", self.next_image)

        self.root.bind("<Return>", self.on_enter_press)
        self.root.bind("<BackSpace>", self.undo)
    
    def init_image(self):
        self.image_path = self.img_root / self.imgs[self.idx]
        self.image = Image.open(self.image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        self.canvas.image = self.photo
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.config(width=self.photo.width(), height=self.photo.height())

        self.image_width, self.image_height = self.photo.width(), self.photo.height()

        # Variables for bounding box
        self.start_x = self.start_y = 0
        self.rect = None

        self.bboxes = []
        # load exisiting bboxes if they exist
        npy_path = self.get_npy_path()
        if os.path.exists(npy_path):
            self.bboxes = [tuple(row) for row in np.load(npy_path)]
            self.draw_bboxes()

    def draw_bboxes(self):
        for item in self.canvas.find_all():
            if self.canvas.type(item) == "rectangle":
                self.canvas.delete(item)

        for x1_norm, y1_norm, x2_norm, y2_norm in self.bboxes:
            x1, y1, x2, y2 = self.get_abs_bbox(x1_norm, y1_norm, x2_norm, y2_norm)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2) 

    def get_normalized_bbox(self, x1, y1, x2, y2):
        return (x1 / self.image_width, y1 / self.image_height, x2 / self.image_width, y2 / self.image_height)
    
    def get_abs_bbox(self, x1_norm, y1_norm, x2_norm, y2_norm):
        return (x1_norm * self.image_width, y1_norm * self.image_height, x2_norm * self.image_width, y2_norm * self.image_height)
    
    def get_npy_path(self):
        name = f"{self.imgs[self.idx].split('.')[0]}_bbox.npy"
        return self.img_root / name
    
    def on_button_press(self, event):
        self.start_x = min(max(event.x, 0), self.image_width)
        self.start_y = min(max(event.y, 0), self.image_height)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        cur_x = min(max(event.x, 0), self.image_width)
        cur_y = min(max(event.y, 0), self.image_height)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x = min(max(event.x, 0), self.image_width)
        end_y = min(max(event.y, 0), self.image_height)
        self.bboxes.append(self.get_normalized_bbox(self.start_x, self.start_y, end_x, end_y))
    
    def next_image(self, event=None):
        if self.idx < len(self.imgs) - 1:
            self.idx += 1
            self.init_image()

    def prev_image(self, event=None):
        if self.idx > 0:
            self.idx -= 1
            self.init_image()
    
    def on_enter_press(self, event=None):
        print(self.bboxes)

        # save bboxes
        npy_path = self.get_npy_path()
        np.save(npy_path, np.array(self.bboxes))

        self.bboxes.clear()
        self.rect = None
        self.next_image()

    def undo(self, event=None):
        if len(self.bboxes) > 1:
            self.bboxes = self.bboxes[:-1]
        else:
            self.bboxes = []
        self.draw_bboxes()

if __name__ == "__main__":
    root = tk.Tk()
    app = BoundingBoxApp(root)
    root.mainloop()
