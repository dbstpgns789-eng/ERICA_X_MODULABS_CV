"""README 그림 만들기: 노트북 출력 그림 꺼내기 + 실험 숫자로 그래프 그리기.

숫자는 노트북 실험 셀 출력에서 옮겨 적은 값이다 (analysis.md와 같음). CPU에서 돌아간다.
"""
import base64
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
                     "axes.edgecolor": "#898781", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, GRAY, ORANGE, RED, INK, SURF = "#2a78d6", "#b5b3ad", "#eb6834", "#d03b3b", "#0b0b0b", "#fcfcfb"
NB = "anomaly_detection_visa_pcb4.ipynb"

# 노트북 셀 코드 표지 → 저장 이름
NB_FIGURES = [
    ("visualize_with_mask(", "sample_with_mask.png"),
    ("plot_roc_curve()", "baseline_roc.png"),
    ("plot_score_distribution()", "baseline_score_distribution.png"),
]

# 원인을 고친 전후 AUROC (README 결과 표). 전 = 같은 조건 재실행 값 (SVM 실험 9 평균, AE 실험 7)
FIXES = [("PaDiM", 0.958, 0.979, "뒤집기 제거 + layer1 추가"),
         ("One-Class SVM", 0.817, 0.872, "뒤집기 제거"),
         ("AutoEncoder", 0.271, 0.928, "Sigmoid 제거")]

# 실험 4·9: 학습 때 좌우 뒤집기, 시드 0·1·2
FLIP = {"PaDiM": {"켬": [0.9632, 0.9530, 0.9574], "끔": [0.9719, 0.9749, 0.9742]},
        "One-Class SVM": {"켬": [0.8152, 0.8180, 0.8171], "끔": [0.8718, 0.8718, 0.8718]}}

# 실험 3: 시험지 고정, 학습 장수만 변경
TRAIN_SIZE = [(50, 0.9536), (200, 0.9641), (502, 0.9674), (804, 0.9719)]

# 실험 10: Gradio 모델의 점수 범위 (최소, 중앙, 최대)
SCORES = {"정상 201장": (12.1, 17.6, 34.1), "불량 100장": (21.1, 31.0, 57.5)}
THRESHOLDS = [(15, "15 (Gradio 코드)"), (23.3, "23.3 (강의)")]


# 실험 7 (20 epoch)의 마지막 loss
AE_LAST_LOSS = [("A 원본", 1.6525, GRAY), ("B Sigmoid 제거", 0.0411, BLUE), ("C Normalize 제거", 0.0026, ORANGE)]


def notebook_ae_losses():
    """셀 [28] 출력의 'Epoch n, Loss: x' 줄에서 원래 AutoEncoder의 epoch별 loss를 읽는다."""
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        text = "".join("".join(o.get("text", "")) for o in c.get("outputs", []))
        found = re.findall(r"Epoch (\d+), Loss: ([\d.]+)", text)
        if found:
            return [int(e) for e, _ in found], [float(v) for _, v in found]
    raise KeyError("AutoEncoder loss 출력을 찾지 못함")


def ae_loss_chart(path="results/ae_loss.png"):
    epochs, losses = notebook_ae_losses()
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURF)
    _ax(ax)
    ax.grid(axis="y", color="#e6e5e1")
    ax.plot(epochs, losses, color=GRAY, lw=2, label="강사 노트북 AE (Sigmoid 있음, 50 epoch)")
    for name, v, color in AE_LAST_LOSS:
        ax.scatter([20], [v], color=color, s=60, zorder=3)
        ax.text(21, v, f"{name}: {v}", va="center", color=INK, fontsize=10)
    ax.set_yscale("log")
    ticks = [0.001, 0.01, 0.1, 1, 2]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (로그 눈금)")
    ax.set_title("Sigmoid가 있으면 loss가 5 epoch 이후 1.6대에서 멈춘다 (점 = 실험 7, 20 epoch 마지막 값)",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def notebook_exp11():
    """실험 11 셀 출력의 'EPOCH n train_loss=.. val_loss=.. test_auroc=..' 줄을 읽는다."""
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "[실험 11]" in "".join(c["source"]):
            text = "".join("".join(o.get("text", "")) for o in c.get("outputs", []))
            rows = re.findall(r"EPOCH\s+(\d+) train_loss=([\d.]+) val_loss=([\d.]+) test_auroc=([\d.]+)", text)
            return [(int(e), float(t), float(v), float(a)) for e, t, v, a in rows]
    raise KeyError("실험 11 출력을 찾지 못함")


def epoch_chart(path="results/ae_epochs.png"):
    rows = notebook_exp11()
    ep = [r[0] for r in rows]
    best_val = min(rows, key=lambda r: r[2])
    best_auc = max(rows, key=lambda r: r[3])
    # loss와 AUROC는 단위가 달라 위아래 두 칸으로 나눈다
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, facecolor=SURF)
    for ax in (a1, a2):
        _ax(ax)
        ax.grid(axis="y", color="#e6e5e1")
    a1.plot(ep, [r[1] for r in rows], color=GRAY, lw=2, label="학습 loss (724장)")
    a1.plot(ep, [r[2] for r in rows], color=BLUE, lw=2, label="검증 loss (정상 80장)")
    a1.scatter([best_val[0]], [best_val[2]], color=BLUE, s=60, zorder=3)
    a1.text(best_val[0], best_val[2] * 1.5, f"최저 {best_val[0]} epoch", ha="center", color=BLUE, fontsize=10)
    a1.set_yscale("log")
    ticks = [0.03, 0.1, 0.3, 0.6]
    a1.set_yticks(ticks, [f"{t:g}" for t in ticks])
    a1.minorticks_off()
    a1.set_ylabel("loss (로그 눈금)")
    a1.legend(frameon=False, loc="upper right")
    a2.plot(ep, [r[3] for r in rows], color=ORANGE, lw=2)
    a2.axvline(best_val[0], color=BLUE, ls="--", lw=1)
    a2.text(best_val[0] + 0.5, 0.45, f"검증 loss로 고른 epoch\nAUROC {best_val[3]:.3f}", color=BLUE, fontsize=9)
    a2.scatter([best_auc[0]], [best_auc[3]], color=ORANGE, s=60, zorder=3)
    a2.text(best_auc[0] - 0.5, best_auc[3] - 0.15, f"시험 최고 {best_auc[0]} epoch {best_auc[3]:.3f}\n(고르면 누수)",
            ha="right", color=INK, fontsize=9)
    a2.set_ylim(0.3, 1.0)
    a2.set_ylabel("시험 AUROC (관찰용)")
    a2.set_xlabel("epoch")
    a1.set_title("고친 AutoEncoder: 20 epoch쯤 AUROC 0.92에 닿고 이후엔 ±0.01 안에서 흔들린다",
                 loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def extract_notebook_figures(out_dir="results"):
    nb = json.load(open(NB, encoding="utf-8"))
    for marker, name in NB_FIGURES:
        # 함수 정의 셀도 표지를 포함하므로, 그림 출력이 있는 첫 셀을 고른다
        png = next(o["data"]["image/png"] for c in nb["cells"]
                   if c["cell_type"] == "code" and marker in "".join(c["source"])
                   for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(base64.b64decode(png))


def _ax(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def fix_chart(path="results/fix_before_after.png"):
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURF)
    _ax(ax)
    ax.grid(axis="x", color="#e6e5e1")
    for i, (name, before, after, what) in enumerate(FIXES):
        ax.plot([before, after], [i, i], color=GRAY, lw=2, zorder=1)
        ax.scatter([before], [i], color=GRAY, s=70, zorder=2)
        ax.scatter([after], [i], color=BLUE, s=70, zorder=2)
        ax.text(before - 0.012, i, f"{before:.3f}", ha="right", va="center", color="#52514e", fontsize=10)
        ax.text(after + 0.012, i, f"{after:.3f}", ha="left", va="center", color=INK, fontsize=10)
        ax.text(min(before, after), i + 0.32, what, color="#52514e", fontsize=9)
    ax.axvline(0.5, color=RED, lw=1, ls="--")
    ax.text(0.505, 2.45, "0.5 = 동전 던지기", color=RED, fontsize=9)
    ax.set_yticks(range(len(FIXES)), [f[0] for f in FIXES])
    ax.set_ylim(-0.5, 2.7)
    ax.invert_yaxis()
    ax.set_xlim(0.15, 1.05)
    ax.set_xlabel("AUROC (회색 = 고치기 전 설정 재실행, 파랑 = 원인을 고친 뒤)")
    ax.set_title("원인을 하나씩 고친 뒤 AUROC", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def flip_chart(path="results/flip_on_off.png"):
    # 두 모델의 눈금 범위가 달라 칸을 나눈다
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), facecolor=SURF)
    for ax, (model, d) in zip(axs, FLIP.items()):
        _ax(ax)
        ax.grid(axis="x", color="#e6e5e1")
        for j, (k, color) in enumerate([("켬", GRAY), ("끔", BLUE)]):
            ax.scatter(d[k], [j] * 3, color=color, s=60, zorder=2)
            m = sum(d[k]) / 3
            same = " (셋 다 같음)" if len(set(d[k])) == 1 else ""
            ax.text(max(d[k]) + 0.003, j, f"평균 {m:.3f}{same}", va="center", color=INK, fontsize=10)
        ax.set_yticks([0, 1], ["뒤집기 켬", "뒤집기 끔"])
        ax.set_ylim(-0.6, 1.6)
        ax.invert_yaxis()
        lo = min(min(v) for v in d.values())
        hi = max(max(v) for v in d.values())
        ax.set_xlim(lo - 0.01, hi + 0.03)
        ax.set_title(f"{model} (점 = 시드 3개)", loc="left", color=INK, fontsize=12)
        ax.set_xlabel("AUROC")
    fig.suptitle("학습 때 좌우 뒤집기를 끄면 두 모델 모두 오른다", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def train_size_chart(path="results/train_size.png"):
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURF)
    _ax(ax)
    ax.grid(axis="y", color="#e6e5e1")
    xs, ys = zip(*TRAIN_SIZE)
    ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=8)
    for x, y in TRAIN_SIZE:
        ax.text(x, y + 0.0015, f"{y:.3f}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(xs)
    ax.set_ylim(0.945, 0.98)
    ax.set_xlabel("학습에 쓴 정상 사진 수 (시험지는 정상 201 + 불량 100 고정)")
    ax.set_ylabel("PaDiM AUROC")
    ax.set_title("학습 장수가 늘면 오르지만 증가 폭은 줄어든다", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def threshold_chart(path="results/threshold_range.png"):
    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=SURF)
    _ax(ax)
    for i, (name, (lo, med, hi)) in enumerate(SCORES.items()):
        color = BLUE if i == 0 else ORANGE
        ax.plot([lo, hi], [i, i], color=color, lw=6, solid_capstyle="round", alpha=0.35)
        ax.scatter([med], [i], color=color, s=80, zorder=3)
        ax.text(med, i - 0.28, f"중앙 {med}", ha="center", color=INK, fontsize=9)
        ax.text(hi + 0.6, i, f"{lo} ~ {hi}", va="center", color="#52514e", fontsize=9)
    for t, label in THRESHOLDS:
        ax.axvline(t, color=RED, lw=1.5, ls="--")
        ax.text(t + 0.3, 1.55, label, color=RED, fontsize=9)
    ax.set_yticks([0, 1], list(SCORES))
    ax.set_ylim(-0.6, 1.9)
    ax.invert_yaxis()
    ax.set_xlim(10, 64)
    ax.set_xlabel("PaDiM 점수 (막대 = 최소~최대, 점 = 중앙값)")
    ax.set_title("판정 기준 15는 정상 점수 중앙값(17.6)보다 낮다", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    extract_notebook_figures()
    ae_loss_chart()
    epoch_chart()
    fix_chart()
    flip_chart()
    train_size_chart()
    threshold_chart()
