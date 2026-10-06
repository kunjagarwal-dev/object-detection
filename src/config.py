from pathlib import Path

# Paths (built from this file's location, so they work from anywhere)
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "sh17"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
LABELS_DIR = PROCESSED_DIR / "labels"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

# Classes (official SH17 list)
CLASS_NAMES = {
    0: "person",
    1: "ear",
    2: "ear-mufs",
    3: "face",
    4: "face-guard",
    5: "face-mask",
    6: "foot",
    7: "tool",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
    16: "safety-vest",
}
NUM_CLASSES = len(CLASS_NAMES)

# Classes to focus on for the demo and per-class analysis
HIGHLIGHT_CLASSES = ["helmet", "safety-vest", "person", "head"]

# Image settings
MAX_SIDE = 1024
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_IMGSZ = 640

# Reproducibility
SEED = 42