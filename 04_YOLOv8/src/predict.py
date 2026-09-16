"""이미지 한 장 또는 폴더에 대해 추론하고 결과를 저장."""
from ultralytics import YOLO


def predict(weights, source, conf=0.25, save_dir="runs/predict"):
    model = YOLO(weights)
    results = model.predict(source=source, conf=conf, save=True, project=save_dir, name="out", exist_ok=True)
    for r in results:
        n = len(r.boxes)
        classes = [model.names[int(c)] for c in r.boxes.cls]
        print(f"{r.path}: {n}개 검출 {classes}")
    return results


if __name__ == "__main__":
    import sys
    weights = sys.argv[1] if len(sys.argv) > 1 else "weights/best.pt"
    source = sys.argv[2] if len(sys.argv) > 2 else "datasets/stamp/test/images"
    predict(weights, source)
