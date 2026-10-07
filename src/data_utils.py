from collections import Counter
from pathlib import Path

from src.config import IMAGES_DIR, LABELS_DIR, IMAGE_EXTS, NUM_CLASSES

def read_label(path):
    """Read one YOLO label file and return a list of box dicts.
    Assumes the file is valid; use validate_label_file to check first."""
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id, x, y, w, h = parts
            boxes.append({
                "class_id": int(class_id),
                "x_center": float(x),
                "y_center": float(y),
                "width": float(w),
                "height": float(h),
            })
    return boxes

def check_pairing():
    image_file = {f.stem for f in IMAGES_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS}
    label_file = {f.stem for f in LABELS_DIR.iterdir() if f.suffix.lower() == ".txt"}

    missing_labels = image_file - label_file
    missing_images = label_file - image_file

    return missing_labels, missing_images

def validate_label_file(path):
    """Return a list of problem descriptions for one label file (empty if valid)."""
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) != 5:
                problems.append(f"line {line_no}: expected 5 values, got {len(parts)}")
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                problems.append(f"line {line_no}: class id '{parts[0]}' is not a whole number")
                continue

            if not 0 <= class_id < NUM_CLASSES:
                problems.append(f"line {line_no}: class id {class_id} out of range")

            try:
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                problems.append(f"line {line_no}: box values are not all numbers")
                continue

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                problems.append(f"line {line_no}: box value outside 0 to 1")
            if w <= 0 or h <= 0:
                problems.append(f"line {line_no}: width or height is not positive")

    return problems


def validate_all_labels(labels_dir=LABELS_DIR):
    """Validate every label file and return a summary dictionary."""
    files_checked = 0
    files_with_problems = 0
    empty_files = 0
    total_boxes = 0
    class_counts = Counter()
    first_problems = []

    for path in sorted(Path(labels_dir).glob("*.txt")):
        files_checked += 1
        problems = validate_label_file(path)

        if problems:
            files_with_problems += 1
            for p in problems:
                if len(first_problems) < 20:
                    first_problems.append(f"{path.name}: {p}")
            continue  # bad files are left out of the box counts

        boxes = read_label(path)
        if not boxes:
            empty_files += 1
        total_boxes += len(boxes)
        class_counts.update(b["class_id"] for b in boxes)

    return {
        "files_checked": files_checked,
        "files_with_problems": files_with_problems,
        "empty_files": empty_files,
        "total_boxes": total_boxes,
        "class_counts": dict(sorted(class_counts.items())),
        "first_problems": first_problems,
    }

