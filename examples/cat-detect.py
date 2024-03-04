from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


def draw_scaled_box(draw: ImageDraw.Draw, box: Tuple[int, int, int, int], scale: float, color: str, width: int) -> None:
    """Draw a scaled bounding box from the center of the original box."""
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    new_width, new_height = (x2 - x1) * scale, (y2 - y1) * scale
    scaled_box = (center_x - new_width / 2, center_y - new_height / 2,
                  center_x + new_width / 2, center_y + new_height / 2)
    draw.rectangle(scaled_box, outline=color, width=width)

def annotate_image(image_path: str, results: List[dict], scale: float = 1.2) -> Image:
    """Annotate the image with bounding boxes."""
    image_pil = Image.open(image_path)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("arial.ttf", 15)

    for result in results:
        box = (result['box']['xmin'], result['box']['ymin'],
               result['box']['xmax'], result['box']['ymax'])
        label = result['label']
        score = result['score']

        # Draw original bounding box
        draw.rectangle(box, outline='red', width=3)

        draw_scaled_box(draw=draw, box=box, scale=1.0, color='maroon', width=1)

        # Draw scaled bounding box
        draw_scaled_box(draw=draw, box=box, scale=scale, color='blue', width=3)

        # Draw label and score
        draw.text((box[0], box[1]), f"{label} {score:.2f}", fill="white", font=font)

    return image_pil


if __name__ == "__main__":
    image_path_out = r"C:\Users\jonma\Pictures\New folder\cat-Screenshot 2024-03-03 085230.png"

    # Allocate a pipeline for object detection
    object_detector = pipeline('object-detection')
    detection_results = object_detector(image_path_out)

    # Annotate and display the image
    annotated_image = annotate_image(image_path_out, detection_results)
    annotated_image.show()
    annotated_image.save(r"C:\Users\jonma\Pictures\New folder\cat-Screenshot 2024-03-03 085230_annotated.png")

    print("Done!")