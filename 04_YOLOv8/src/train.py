"""YOLOv8n 학습. Roboflow stamp 데이터셋(shoes, stamp 2클래스)."""
from ultralytics import YOLO


def train(data_yaml, epochs=20, imgsz=640, batch=16, name="stamp_detect", project="runs/detect"):
    model = YOLO("yolov8n.pt")
    return model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project=project,
        device=0,
    )


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "datasets/stamp/data.yaml"
    train(data)
