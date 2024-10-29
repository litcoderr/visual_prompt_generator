"""
This is an example code that utilizes VisualPromptGenerator to generate visual reasoning dataset
"""
import os

from PIL import Image
from pathlib import Path

from vpgen import VisualPromptGenerator


imgs_root = Path("/Users/jameschee/Desktop/sample_data/small_dataset/images")
img_types = os.listdir(imgs_root)

def get_img_path(typ: int, idx: int):
    sub_root = imgs_root / img_types[typ]
    imgs = os.listdir(sub_root)
    return sub_root / imgs[idx]


if __name__ == "__main__":
    # initialize visual prompt generator
    key_path = Path("/Users/jameschee/Desktop") / "openai_key.txt"
    vpgen = VisualPromptGenerator(key_path=key_path)

    # read sample image
    img = Image.open(get_img_path(0, 25))
    img.show()

    bboxes = vpgen.extract_bounding_boxes(img, additional_prompt=f"This image type is {img_types[0]}.")
    bboxed_img = vpgen.draw_bounding_boxes(img, bboxes)
    bboxed_img.show()
