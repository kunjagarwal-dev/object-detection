from ultralytics import YOLO
from src.config import ROOT

def export_to_oonx():
    model = YOLO(ROOT / "models" / "final_yolo11n_960.pt")
    model.export(
        format="onnx",
        imgsz=960,
        simplify=True,
        dynamic=False
    )

def val_onnx():
    model = YOLO(ROOT / "models" / "final_yolo11n_960.onnx")
    return model.val(
        data=str(ROOT / "configs" / "data.yaml"),
        split="val",
        imgsz=960,
        batch=1,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs" / "val"),
        name="onnx_cpu_960",
    )

if __name__ == "__main__":
    val_onnx()