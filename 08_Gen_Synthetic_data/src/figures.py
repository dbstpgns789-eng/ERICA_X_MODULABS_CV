"""README 그림 만들기: A2 그래프, C1 GPU 메모리 그래프, AnomalyAny 비교 그림과 4장 모음.

숫자는 노트북 셀 출력에서 옮겨 적은 값이다. CPU에서 돌아간다.
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
                     "axes.edgecolor": "#898781", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, GRAY, RED, INK, SURF = "#2a78d6", "#b5b3ad", "#d03b3b", "#0b0b0b", "#fcfcfb"
GB_TO_GIB = 1e9 / 2**30

# 실험 A2 셀 출력
A2 = {"fp32": (19.44, 6.33), "fp16": (5.01, 3.42)}  # (초, GB)

# 8-1은 A2의 PyTorch 최대 할당(max_memory_allocated/1e9, GB),
# 8-2는 OOM 메시지의 "this process has ... in use"(GiB). T4 한계와 비교할 값은 프로세스 사용량이다
T4_GIB = 14.56
MEMORY = [  # (이름, 사용 GiB, 실패한 추가 요청 GiB, 설명)
    ("8-1 SD 생성 · fp16", 3.42 * GB_TO_GIB, 0, "PyTorch 최대 할당, 성공"),
    ("8-1 SD 생성 · fp32", 6.33 * GB_TO_GIB, 0, "PyTorch 최대 할당, 성공"),
    ("AnomalyAny · fp32 (강사 원본)", 14.16, 1.0, "프로세스 사용 + 1GiB 요청 실패 → OOM"),
    ("AnomalyAny · fp32 + autocast", 14.43, 0.5, "프로세스 사용 + 0.5GiB 요청 실패 → OOM"),
    ("AnomalyAny · fp16 로드", 2.16 * GB_TO_GIB, 0, "로드 직후 할당, 생성에서 dtype 불일치 에러"),
]


def _clean(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", lw=1)
    ax.set_axisbelow(True)


def a2_chart(path="results/a2_chart.png"):
    # 단위가 다른 두 값이라 축 하나에 겹치지 않고 그래프를 나눈다
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), facecolor=SURF)
    for ax, (title, idx, unit) in zip(axs, [("생성 시간 (30단계)", 0, "초"), ("최대 GPU 메모리", 1, "GB")]):
        _clean(ax)
        vals = [A2["fp32"][idx], A2["fp16"][idx]]
        bars = ax.barh(["fp32", "fp16"], vals, color=[GRAY, BLUE], height=0.5)
        for r, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.02, r.get_y() + r.get_height() / 2, f"{v:g}{unit}", va="center", color=INK)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", color=INK, fontsize=12)
    fig.suptitle("실험 A2. fp16: 시간 3.9배 단축, 메모리 54%  (그림은 거의 같음)",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def memory_chart(path="results/c1_gpu_memory.png"):
    labels = [m[0] for m in MEMORY]
    used = [m[1] for m in MEMORY]
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURF)
    _clean(ax)
    ax.barh(labels, used, color=[BLUE, BLUE, GRAY, GRAY, GRAY], height=0.55)
    for i, (_, u, req, note) in enumerate(MEMORY):
        if req:
            ax.barh(i, req, left=u, color="white", edgecolor=RED, hatch="///", height=0.55, lw=1)
        inside = u > 12
        ax.text(0.25 if inside else u + 0.25, i, f"{u:.2f}GiB  {note}", va="center",
                color="white" if inside else INK, fontsize=10)
    ax.axvline(T4_GIB, color=RED, lw=2)
    ax.text(T4_GIB - 0.16, 4, f"T4 한계 {T4_GIB}GiB", color=RED, ha="right", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 16.5)
    ax.set_xlabel("GPU 메모리 (GiB)")
    ax.set_title("같은 SD 1.5인데 AnomalyAny만 T4를 넘는다\n(추정 원인: 생성 중 역전파 + fp32 로드 + CLIP 추가 로드)",
                 loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)


def anomalyany_compare(original, generated, attention, path="results/c_anomalyany_vs_original.png",
                       zoom_box=(110, 160, 230, 310)):
    """원본 / AnomalyAny 결과 / crack attention을 나란히, 아래에 커넥터 확대."""
    o = original.convert("RGB").resize((512, 512))
    g = generated.convert("RGB").resize((512, 512))
    a = attention.convert("RGB").crop((0, 0, 256, 256)).resize((512, 512))  # 아래 글자 'crack' 잘라냄
    canvas = Image.new("RGB", (512 * 3 + 40, 512 + 340), "white")
    d = ImageDraw.Draw(canvas)
    for k, (im, t) in enumerate([(o, "1. original"), (g, "2. AnomalyAny result"), (a, "3. crack attention")]):
        canvas.paste(im, (k * 532, 40))
        d.text((k * 532 + 10, 10), t, fill="black")
    for k, (im, t) in enumerate([(o, "original zoom (left connector)"), (g, "result zoom (left connector)")]):
        canvas.paste(im.crop(zoom_box).resize((240, 300)), (k * 260 + 10, 590))
        d.text((k * 260 + 10, 570), t, fill="black")
    canvas.save(path)


def anomalyany_grid(pairs, path="results/c_anomalyany_4seeds.png"):
    """강사 출력 4장: 윗줄 생성 결과, 아랫줄 crack attention."""
    canvas = Image.new("RGB", (400 * len(pairs), 800), "white")
    for i, (gen, att) in enumerate(pairs):
        canvas.paste(gen.convert("RGB").resize((400, 400)), (i * 400, 0))
        canvas.paste(att.convert("RGB").crop((0, 0, 256, 256)).resize((400, 400)), (i * 400, 400))
    canvas.save(path)


if __name__ == "__main__":
    import sys
    from extract_outputs import anomalyany_pcb_images
    a2_chart()
    memory_chart()
    # 원본 PCB 사진 경로를 인자로 받는다 (강사 실습 자료라 레포에는 없음)
    imgs = [Image.open(io.BytesIO(b)) for b in anomalyany_pcb_images()]
    anomalyany_grid(list(zip(imgs[0::2], imgs[1::2])))
    if len(sys.argv) > 1:
        anomalyany_compare(Image.open(sys.argv[1]), imgs[0], imgs[1])
