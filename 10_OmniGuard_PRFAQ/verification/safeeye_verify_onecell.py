import subprocess, sys
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv"],capture_output=True,text=True).stdout)
subprocess.run([sys.executable,"-m","pip","-q","install","rfdetr","rtmlib","onnxruntime","transformers","huggingface_hub","opencv-python-headless"],check=True)
# rfdetr rtmlib onnxruntime transformers huggingface_hub opencv-python-headless


import json, time, random, os
import numpy as np, cv2, torch
from PIL import Image
from huggingface_hub import hf_hub_download
REPO = "Voxel51/Safe_and_Unsafe_Behaviours"
random.seed(0)
samples = json.load(open(hf_hub_download(REPO, "samples.json", repo_type="dataset")))
samples = samples["samples"] if isinstance(samples, dict) else samples
PLAN = {"Unauthorized Intervention": 8, "Authorized Intervention": 6, "Safe Walkway": 6}
picked = []
for label, n in PLAN.items():
    pool = [s for s in samples if s["ground_truth"]["label"] == label]
    picked += [(label, s["filepath"]) for s in random.sample(pool, n)]
FRAMES = []  # (label, clip, idx, bgr)
for label, fp in picked:
    path = hf_hub_download(REPO, fp, repo_type="dataset")
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i, f in enumerate(np.linspace(0, n - 1, 5).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if ok: FRAMES.append((label, os.path.basename(fp), i, img))
    cap.release()
print(len(picked), "clips,", len(FRAMES), "frames", FRAMES[0][3].shape)


# 1. 사람 탐지: RF-DETR-Small
from rfdetr import RFDETRSmall
try:
    from rfdetr.util.coco_classes import COCO_CLASSES
except Exception:
    COCO_CLASSES = {1: "person"}
det = RFDETRSmall()
_ = det.predict(Image.fromarray(cv2.cvtColor(FRAMES[0][3], cv2.COLOR_BGR2RGB)), threshold=0.5)  # warmup
person, t_det = [], []
for label, clip, i, img in FRAMES:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter(); d = det.predict(pil, threshold=0.5)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_det.append((time.perf_counter() - t0) * 1000)
    names = [COCO_CLASSES.get(int(c), str(c)) for c in d.class_id]
    person.append(sum(n == "person" for n in names))
print("RF-DETR ms/frame median %.1f" % np.median(t_det))


# 2. 숙임 자세: RTMPose (rtmlib Body = 사람 탐지 + RTMPose, CPU onnxruntime)
from rtmlib import Body
body = Body(mode="balanced", backend="onnxruntime", device="cpu")
bent, t_pose = [], []
def torso_angle(k, s):
    sh, hp = (k[5] + k[6]) / 2, (k[11] + k[12]) / 2
    if min(s[5], s[6], s[11], s[12]) < 0.3: return None
    v = sh - hp
    return float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))
for label, clip, i, img in FRAMES:
    t0 = time.perf_counter(); kps, sc = body(img); t_pose.append((time.perf_counter() - t0) * 1000)
    angs = [a for a in (torso_angle(k, s) for k, s in zip(kps, sc)) if a is not None]
    bent.append(int(any(a >= 45 for a in angs)))
print("RTMPose(CPU) ms/frame median %.1f" % np.median(t_pose))


# 3. 도구 탐지(학습 없음): Grounding DINO tiny
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
gid = "IDEA-Research/grounding-dino-tiny"
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoProcessor.from_pretrained(gid); gd = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to(dev)
TEXT = "spray can. wrench. tool."
tool, tool_names, t_gd = [], [], []
for label, clip, i, img in FRAMES:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inp = proc(images=pil, text=TEXT, return_tensors="pt").to(dev)
    t0 = time.perf_counter()
    with torch.no_grad(): out = gd(**inp)
    t_gd.append((time.perf_counter() - t0) * 1000)
    try:
        r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.35, text_threshold=0.25, target_sizes=[pil.size[::-1]])[0]
    except TypeError:
        r = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.35, text_threshold=0.25, target_sizes=[pil.size[::-1]])[0]
    labs = r.get("text_labels", r.get("labels", []))
    tool.append(len(labs)); tool_names.append([str(x) for x in labs])
print("GroundingDINO ms/frame median %.1f" % np.median(t_gd))


# 4. 집계
rows = {}
for k, (label, clip, i, img) in enumerate(FRAMES):
    r = rows.setdefault(label, {"frames": 0, "person_frames": 0, "bent_frames": 0, "tool_frames": 0})
    r["frames"] += 1; r["person_frames"] += person[k] > 0; r["bent_frames"] += bent[k]; r["tool_frames"] += tool[k] > 0
from collections import Counter
summary = {
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "frame_size": list(FRAMES[0][3].shape[:2]),
    "clips": len(picked), "frames": len(FRAMES), "by_class": rows,
    "tool_label_counts": dict(Counter(n for ns in tool_names for n in ns)),
    "ms_median": {"rfdetr_s_gpu_pytorch": round(float(np.median(t_det)), 1),
                  "rtmpose_body_cpu": round(float(np.median(t_pose)), 1),
                  "groundingdino_tiny_gpu": round(float(np.median(t_gd)), 1)},
}
print("===RESULT===")
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("===END===")
json.dump(summary, open("safeeye_verify_result.json", "w"), ensure_ascii=False, indent=1)
