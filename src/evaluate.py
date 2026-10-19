from ultralytics import YOLO
from src.config import ROOT

def validate_640_at_960():
    model = YOLO(str(ROOT / "runs" / "train" / "v11_640" / "weights" / "best.pt"))
    result = model.val(
        data= str(ROOT / "configs" / "data.yaml"),
        imgsz=960,
        batch=16,
        workers = 2,
        device=0,
        name="v11_640_at_960",
        project=str(ROOT / "runs" / "val")
    )

def validate_640_at_640():
    model = YOLO(str(ROOT / "runs" / "train" / "v11_640" / "weights" / "best.pt"))
    result = model.val(
        data= str(ROOT / "configs" / "data.yaml"),
        imgsz=640,
        batch=16,
        workers = 2,
        device=0,
        name="v11_640_at_640",
        project=str(ROOT / "runs" / "val")
    )

if __name__ == "__main__":
    validate_640_at_640()