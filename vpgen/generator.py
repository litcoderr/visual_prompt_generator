from typing import List, Tuple

import re
import io
import openai
import base64

from PIL import Image, ImageDraw


class VisualPromptGenerator:
    def __init__(self, key_path: str):
        """
        Args:
            key_path: text file path consisting of oepnai api key
        """
        with open(key_path, "r") as f:
            self.openai_key = f.readline().rstrip()

    def preprocess_img(self, img: Image):
        """
        Converts a PIL Image to base64 byte string for openai api input
        """
        with io.BytesIO() as output:
            img.save(output, format='PNG')
            binary = output.getvalue()

        return base64.b64encode(binary).decode('utf-8')

    def draw_bounding_boxes(self, img: Image, bboxes: List[Tuple[float, float, float, float]], color="red", width=3) -> Image:
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        
        for box in bboxes:
            x_min = int(box[0] * img_width)
            y_min = int(box[1] * img_height)
            x_max = int(box[2] * img_width)
            y_max = int(box[3] * img_height)
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
        
        return img

    def extract_bounding_boxes(self, img: Image, additional_prompt=None) -> List[Tuple[float, float, float, float]]:
        """
        Args:
            img: PIL Image
        Returns:
            List of tuples that represents a bounding box (x1, x2, y1, y2). ex. [(0.1, 0.2, 0.3, 0.4), ...]
        """
        client = openai.OpenAI(api_key=self.openai_key)        

        system_prompt = """
        You are a professional bounding box generator. You see objects in a given image and generate bounding boxes. You only reply in following format.
        
        [x1, x2, y1, y2], [x1, x2, y1, y2], ... , [x1, x2, y1, y2]

        top-left of image is (0,0) and bottom-right of image is (1,1). x-axis is the horizontal axis and y-axis is vertical axis.
        each coordinate is a normalized floating point number ranging from 0 to 1. Be as precise as possible.
        """        
        user_prompt = """
        Generate bounding boxes. Be very very precise with generating bounding boxes.
        """        
        if additional_prompt is not None:
            user_prompt += f"\n{additional_prompt}"

        try:
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/jpeg;base64,{self.preprocess_img(img)}"
                            },
                        },
                    ],
                }
            ])
        except Exception as e:
            print(f"[openai] couldn't retrieve generated result {e}")
            return None

        # extract reward from response
        content = response.choices[0].message.content
        boxes = re.findall(r"\[\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*\]", content)
        boxes = [tuple(map(float, box)) for box in boxes]
        return boxes
