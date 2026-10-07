from src.config import IMAGES_DIR, LABELS_DIR, CLASS_NAMES, NUM_CLASSES
from src.data_utils import read_label

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def _make_palette(n):
    """Build n clearly different BGR colours. The golden-ratio step keeps
    neighbouring class IDs from getting similar hues."""
    palette = []
    for i in range(n):
        hue = int(180 * ((i * 0.618) % 1))  # OpenCV hue range is 0-179
        hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append(tuple(int(c) for c in bgr))
    return palette


COLORS = _make_palette(NUM_CLASSES)

def draw_boxes(image_path, label_path):
    """Draw boxes on an image from a YOLO label file and return the image."""
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Read the label file
    boxes = read_label(label_path)

    # Draw each box
    for box in boxes:
        class_id = box["class_id"]
        x_center = box["x_center"]
        y_center = box["y_center"]
        width = box["width"]
        height = box["height"]

        # Convert normalized coordinates to pixel coordinates
        img_height, img_width = image.shape[:2]
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # Draw the rectangle and label
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = CLASS_NAMES.get(class_id, "unknown")
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_grid(image_names, rows, cols, images_dir=IMAGES_DIR, labels_dir=LABELS_DIR):
    """Show images with their boxes in a rows x cols grid."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, image_names):
        image_path = Path(images_dir) / name
        label_path = Path(labels_dir) / (image_path.stem + ".txt")
        ax.imshow(draw_boxes(image_path, label_path))
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    for ax in axes[len(image_names):]:  # hide unused cells
        ax.axis("off")

    plt.tight_layout()
    plt.show()

