"""Single-image inference used by the Streamlit app and by notebooks."""
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.config import MODELS_DIR

IMGSZ = 960  # the ONNX model is exported static at 960, so never change this
ONNX_PATH = MODELS_DIR / "final_yolo11n_960.onnx"
PT_PATH = MODELS_DIR / "final_yolo11n_960.pt"


def load_model(path: Path | None = None) -> YOLO:
    """Load the ONNX model if present, otherwise the PyTorch weights."""
    path = Path(path) if path else (ONNX_PATH if ONNX_PATH.exists() else PT_PATH)
    return YOLO(str(path), task="detect")


def predict(model: YOLO, image_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.7):
    """Run detection on one RGB image.

    Returns (annotated RGB image, Counter of class name -> count, list of detections).
    """
    result = next(
        iter(
            model.predict(
                image_rgb,
                imgsz=IMGSZ,
                conf=conf,
                iou=iou,
                device="cpu",
                verbose=False,
                stream=False,
            )
        ),
        None,
    )
    if not isinstance(result, Results):
        raise RuntimeError("YOLO did not return a detection result.")

    boxes = result.boxes
    if boxes is None:
        raise RuntimeError("YOLO detection result has no bounding boxes.")

    annotated = result.plot()[:, :, ::-1]  # plot() returns BGR, convert to RGB

    detections = []
    for box, cls, score in zip(
        boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
    ):
        detections.append(
            {"class": result.names[int(cls)], "confidence": round(score, 3),
             "box_xyxy": [round(v, 1) for v in box]}
        )
    counts = Counter(d["class"] for d in detections)
    return annotated, counts, detections