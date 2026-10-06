from pathlib import Path
import shutil

import cv2
from tqdm import tqdm

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resize_images(input_dir, output_dir, max_side=1024, limit=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    if not output_path.exists():
        output_path.mkdir(parents=True)

    files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXT)
    if limit:
        files = files[:limit]

    processed, skip, failed = 0,0,[]

    for image_file in tqdm(files, desc="Resizing images"):
        output_file = output_path / image_file.name

        if output_file.exists():
            skip += 1
            print(f"Warning, file already exists: {output_file}")
            continue

        img = cv2.imread(str(image_file))
        if img is None:
            failed.append(image_file)
            print(f"Warning: Could not read image {image_file}")
            continue

        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            resized_img = img
        else:
            scale = max_side / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(output_file), resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed += 1

    print(f"Processed {processed} images, skipped {skip}, failed {len(failed)}")

if __name__ == "__main__":
    resize_images(
        PROJECT_ROOT / "data" / "raw" / "sh17" / "images",
        PROJECT_ROOT / "data" / "processed" / "images",
    )
