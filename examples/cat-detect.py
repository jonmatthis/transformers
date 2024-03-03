import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# Load an image from disk
image_path = r"C:\Users\jonma\Pictures\New folder\cat-Screenshot 2024-03-03 085230.png"
image_pil = Image.open(image_path)

# Allocate a pipeline for object detection
object_detector = pipeline('object-detection')
results = object_detector(image_pil)

# Draw bounding boxes and probabilities
draw = ImageDraw.Draw(image_pil)
for result in results:
    box = result['box']
    label = result['label']
    score = result['score']
    x1, y1, x2, y2 = box.values()

    # Draw the rectangle
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Optional: Draw label and score
    draw.text((x1, y1), f"{label} {score:.2f}", fill="white", font=ImageFont.truetype("arial.ttf", 15))

image_pil.show()