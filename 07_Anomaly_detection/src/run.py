"""PaDiM, AutoEncoder, One-Class SVM을 같은 데이터로 학습·평가하고 PaDiM을 pickle로 저장.

노트북 셀 [27]~[28]을 옮긴 것. 사용법: python src/run.py ./VisA pcb4
"""
import os
import pickle

import torch

from dataset import create_visa_data_loaders
from evaluate import evaluate
from models import FeatureExtractor, PaDiM, AutoEncoderAnomalyDetector, OneClassSVMAnomalyDetector


def run(data_path, category, epochs=50, pretrained=True, layers=("layer1", "layer2", "layer3"),
        out_path="padim_outputs/padim_model_state.pkl", seed=0):
    # PaDiM 무작위 투영 때문에 시드가 없으면 돌릴 때마다 ±0.002 정도 흔들린다 (실험 3)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    train_loader, test_loader = create_visa_data_loaders(data_path, category, batch_size=16, train_ratio=0.8)
    test_labels = test_loader.dataset.labels
    print(f"훈련 {len(train_loader.dataset)}장 / 테스트 {len(test_loader.dataset)}장 (불량 {sum(test_labels)}장)")

    # 노트북 셀 [28]은 layer2+3, Gradio 셀은 layer1+2+3이라 저장한 모델을 바꿔 쓸 수 없었다.
    # 실험 5에서 더 좋았던 layer1+2+3으로 통일
    feature_extractor = FeatureExtractor('wide_resnet50', list(layers), pretrained=pretrained).to(device)
    results = {}

    print("\n=== PaDiM 실행 ===")
    padim = PaDiM(feature_extractor, device, d_reduced=100)
    padim.fit(train_loader)
    scores, _ = padim.predict(test_loader)
    results['PaDiM'] = evaluate(test_labels, scores, 'PaDiM')

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(padim.state(), f)
    print(f"PaDiM 저장: {out_path}")

    print("\n=== AutoEncoder 실행 ===")
    ae = AutoEncoderAnomalyDetector(device)
    ae.fit(train_loader, epochs=epochs)
    results['AutoEncoder'] = evaluate(test_labels, ae.predict(test_loader), 'AutoEncoder')

    print("\n=== One-Class SVM 실행 ===")
    ocsvm = OneClassSVMAnomalyDetector(feature_extractor, device, nu=0.1)
    ocsvm.fit(train_loader)
    results['One-Class SVM'] = evaluate(test_labels, ocsvm.predict(test_loader), 'One-Class SVM')

    print(f"{'방법':15s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s}")
    for name, r in results.items():
        print(f"{name:15s} {r['AUROC']:7.4f} {r['AUPRC']:7.4f} {r['Best_F1']:7.4f}")
    return results


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "./VisA"
    category = sys.argv[2] if len(sys.argv) > 2 else "pcb4"
    run(data, category)
