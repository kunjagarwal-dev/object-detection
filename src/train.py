from ultralytics import YOLO
from src.config import ROOT

def smoke_train():
    model = YOLO("yolov8n.pt")
    result = model.train(data= str(ROOT / "configs" / "data.yaml"),
                         epochs=1,
                         imgsz=640,
                         batch=16,
                         fraction=0.05,
                         workers = 2,
                         device=0,
                         name="smoke",
                         project="runs")

    return result

def train_baseline():
    model = YOLO("yolov8n.pt")
    result = model.train(data= str(ROOT / "configs" / "data.yaml"),
                            epochs=60,
                            patience=15,
                            imgsz=640,
                            batch=16,
                            workers = 2,
                            device=0,
                            name="baseline_v8n_640",
                            project=str(ROOT / "runs" / "train"))

    return result

if __name__ == "__main__":
    train_baseline()