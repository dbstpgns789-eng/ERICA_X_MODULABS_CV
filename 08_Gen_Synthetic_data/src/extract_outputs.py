"""노트북 셀 출력에 남은 그림을 results/로 꺼낸다.

셀 번호는 마크다운 셀을 넣으면 밀리므로, 셀 코드의 첫 부분(표지 문자열)으로 찾는다.
"""
import base64
import json
import os

SD_NB = "stable_diffusion_inpainting_pcb.ipynb"
AA_NB = "anomalyany_pcb_crack.ipynb"

# (노트북, 셀 코드에 들어 있는 표지, 몇 번째 그림, 저장 이름)
TARGETS = [
    (SD_NB, "inpainted_defect_", 2, "gray_plate_dent.png"),
    (SD_NB, "treasure chest", 2, "beach_chest.png"),
    (SD_NB, "[실험 A1]", 0, "a1_guidance.png"),
    (SD_NB, "[실험 A2]", 0, "a2_fp32_vs_fp16.png"),
    (SD_NB, "[실험 B1 재시도]", 0, "b1_pcb_inpaint.png"),
    (SD_NB, "[실험 B1 재시도]", 1, "b1_text_zoom.png"),
    (SD_NB, "[실험 B2]", 0, "b2_composite.png"),
    (SD_NB, "[실험 A3]", 0, "a3_steps.png"),
]


def cell_images(nb_path, marker):
    """표지가 들어 있는 첫 코드 셀의 png 출력들을 bytes 목록으로 돌려준다."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return [base64.b64decode(o["data"]["image/png"])
                    for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    raise KeyError(f"{nb_path}: '{marker}' 셀을 찾지 못함")


def anomalyany_pcb_images(nb_path=AA_NB):
    """AnomalyAny PCB 생성 셀의 출력: [생성 1, attention 1, 생성 2, attention 2, ...]."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(code):
        if 'type = "pcb"' in "".join(c["source"]):
            nxt = code[i + 1]
            return [base64.b64decode(o["data"]["image/png"])
                    for o in nxt.get("outputs", []) if "image/png" in o.get("data", {})]
    raise KeyError("PCB 생성 셀을 찾지 못함")


def extract_all(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    for nb, marker, k, name in TARGETS:
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(cell_images(nb, marker)[k])
        print("saved:", name)


if __name__ == "__main__":
    extract_all()
