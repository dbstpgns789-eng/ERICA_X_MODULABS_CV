"""이상 점수 평가: AUROC, AUPRC, best F1과 정상 점수 기반 임계값. 노트북 셀 [26]을 옮긴 것.

노트북의 warnings.filterwarnings('ignore')는 옮기지 않았다.
실험 1에서 시험 세트에 정상이 0장일 때 AUROC가 nan이 됐는데 그 경고를 숨기고 있었다.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def evaluate(y_true, y_scores, method_name):
    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    valid = ~np.isnan(f1_scores)
    if np.any(valid):
        valid_f1 = f1_scores[valid]
        valid_th = thresholds[valid] if len(thresholds) == len(f1_scores) else thresholds[:len(valid_f1)]
        best_idx = np.argmax(valid_f1)
        best_f1 = valid_f1[best_idx]
        best_threshold = valid_th[best_idx] if len(valid_th) > best_idx else 0.5
    else:
        best_f1 = 0.0
        best_threshold = 0.5

    print(f"{method_name} 결과:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  Best F1: {best_f1:.4f}")
    print()
    return {'AUROC': auroc, 'AUPRC': auprc, 'Best_F1': best_f1, 'Best_Threshold': best_threshold}


def threshold_from_normal(normal_scores, q=99):
    """정상 점수의 q 퍼센타일을 임계값으로.

    best F1 임계값은 시험 세트로 고르고 같은 세트로 성적을 내서 낙관적이다 (실험 10).
    불량 사진은 모으기 어려우니, 학습에 안 쓴 정상 사진의 점수 분포로 정한다.
    """
    return float(np.percentile(normal_scores, q))

