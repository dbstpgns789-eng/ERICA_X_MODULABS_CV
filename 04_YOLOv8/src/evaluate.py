"""학습된 가중치로 valid 또는 test 세트 평가. 클래스별 P/R/mAP 출력."""
from ultralytics import YOLO


def evaluate(weights, data_yaml, split="val", name=None, project="runs/detect"):
    model = YOLO(weights)
    m = model.val(data=data_yaml, split=split, name=name or f"{split}_eval", project=project)
    print(f"[{split}] mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  "
          f"P={m.box.mp:.3f}  R={m.box.mr:.3f}")
    for i, cls in m.names.items():
        print(f"  {cls:6s} mAP50-95={m.box.maps[i]:.3f}")
    return m


if __name__ == "__main__":
    import sys
    weights = sys.argv[1] if len(sys.argv) > 1 else "weights/best.pt"
    data = sys.argv[2] if len(sys.argv) > 2 else "datasets/stamp/data.yaml"
    split = sys.argv[3] if len(sys.argv) > 3 else "val"
    evaluate(weights, data, split)
