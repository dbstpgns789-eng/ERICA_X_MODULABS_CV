import subprocess, sys
subprocess.run([sys.executable,"-m","pip","-q","install","rfdetr","rtmlib","onnxruntime","transformers","huggingface_hub","opencv-python-headless"],check=True)
import json, time, random, os
import numpy as np, cv2, torch
from PIL import Image
from huggingface_hub import hf_hub_download
import transformers
print("transformers", transformers.__version__)
REPO = "Voxel51/Safe_and_Unsafe_Behaviours"
random.seed(0)
samples = json.load(open(hf_hub_download(REPO, "samples.json", repo_type="dataset")))
samples = samples["samples"] if isinstance(samples, dict) else samples
PLAN = {"Unauthorized Intervention": 8, "Authorized Intervention": 6, "Safe Walkway": 6}
picked = []
for label, n in PLAN.items():
    pool = [s for s in samples if s["ground_truth"]["label"] == label]
    picked += [(label, s["filepath"]) for s in random.sample(pool, n)]
FRAMES = []
for label, fp in picked:
    path = hf_hub_download(REPO, fp, repo_type="dataset")
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i, f in enumerate(np.linspace(0, n - 1, 5).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if ok: FRAMES.append((label, os.path.basename(fp), i, img))
    cap.release()
print(len(FRAMES), "frames")

# A. RF-DETR: person boxes + optimized timing
from rfdetr import RFDETRSmall
try:
    from rfdetr.utilities.coco_classes import COCO_CLASSES
except Exception:
    import rfdetr.utilities as _u, pkgutil
    print("rfdetr.utilities modules:", [m.name for m in pkgutil.iter_modules(_u.__path__)])
    COCO_CLASSES = None
if COCO_CLASSES is None:
    COCO_CLASSES = {1: "person"}
print("COCO_CLASSES[1] =", COCO_CLASSES.get(1), "| [0] =", COCO_CLASSES.get(0))
det = RFDETRSmall()
pboxes, t_raw = [], []
for _, _, _, img in FRAMES:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize(); t0 = time.perf_counter(); d = det.predict(pil, threshold=0.5); torch.cuda.synchronize()
    t_raw.append((time.perf_counter()-t0)*1000)
    pboxes.append([b for b, c in zip(d.xyxy, d.class_id) if COCO_CLASSES.get(int(c)) == "person"])
t_opt = []
try:
    det.optimize_for_inference()
    for _, _, _, img in FRAMES[:30]:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        torch.cuda.synchronize(); t0 = time.perf_counter(); det.predict(pil, threshold=0.5); torch.cuda.synchronize()
        t_opt.append((time.perf_counter()-t0)*1000)
except Exception as e:
    print("optimize_for_inference failed:", repr(e)[:200])
heights = [float(b[3]-b[1]) for bs in pboxes for b in bs]
print("person box height px: median %.0f, min %.0f, max %.0f" % (np.median(heights), min(heights), max(heights)))

# B. RTMPose diagnostics: keypoint scores and torso angles
from rtmlib import Body
body = Body(mode="balanced", backend="onnxruntime", device="cpu")
angles, kp_scores, n_people, poses = [], [], 0, []
for _, _, _, img in FRAMES:
    kps, sc = body(img); poses.append((kps, sc))
    for k, s in zip(kps, sc):
        n_people += 1
        kp_scores.append(float(np.mean([s[5], s[6], s[11], s[12]])))
        sh, hp = (k[5]+k[6])/2, (k[11]+k[12])/2; v = sh - hp
        angles.append(float(np.degrees(np.arctan2(abs(v[0]), abs(v[1])+1e-6))))
angles, kp_scores = np.array(angles), np.array(kp_scores)
print("pose people", n_people, "torso kp score median %.2f, share>=0.3: %.2f" % (np.median(kp_scores), (kp_scores>=0.3).mean()))
print("torso angle (deg) percentiles 50/90/99/max:", np.round(np.percentile(angles,[50,90,99,100]),1).tolist(), "count>=30:", int((angles>=30).sum()), "count>=45:", int((angles>=45).sum()))

# C. Grounding DINO diagnostics
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
gid = "IDEA-Research/grounding-dino-tiny"
proc = AutoProcessor.from_pretrained(gid); gd = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to("cuda")
TEXT = "a spray can. a wrench. a hand tool."
tool_frames, label_counter, scores_all, gboxes = 0, {}, [], []
from collections import Counter
label_counter = Counter()
for _, _, _, img in FRAMES:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inp = proc(images=pil, text=TEXT, return_tensors="pt").to("cuda")
    with torch.no_grad(): out = gd(**inp)
    r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.35, text_threshold=0.25, target_sizes=[pil.size[::-1]])[0]
    labs = [str(x) for x in r.get("text_labels", r.get("labels", []))]
    sc = [float(x) for x in r["scores"]]
    keep = [(l, s, b) for l, s, b in zip(labs, sc, r["boxes"].tolist()) if l.strip()]
    scores_all += sc; label_counter.update(l for l, _, _ in keep)
    tool_frames += len(keep) > 0; gboxes.append(keep)
print("GDINO keys:", list(r.keys()))
print("GDINO frames with non-empty labeled box:", tool_frames, "/", len(FRAMES), "labels:", dict(label_counter), "score median %.2f" % (np.median(scores_all) if scores_all else -1))

# D. contact sheet (6 frames: 4 intervention, 2 walkway)
idx = [0, 10, 20, 45, 75, 90]
tiles = []
for k in idx:
    img = FRAMES[k][3].copy()
    for b in pboxes[k]: cv2.rectangle(img, tuple(map(int,b[:2])), tuple(map(int,b[2:])), (0,255,0), 3)
    for l, s, b in gboxes[k]: cv2.rectangle(img, tuple(map(int,b[:2])), tuple(map(int,b[2:])), (0,0,255), 3); cv2.putText(img, f"{l} {s:.2f}", (int(b[0]), int(b[1])-8), 0, 1.0, (0,0,255), 2)
    kps, sc = poses[k]
    for kp, s in zip(kps, sc):
        for (x, y), ss in zip(kp, s):
            if ss >= 0.3: cv2.circle(img, (int(x), int(y)), 5, (255,0,0), -1)
    cv2.putText(img, FRAMES[k][0], (20, 50), 0, 1.5, (255,255,255), 4)
    tiles.append(cv2.resize(img, (640, 360)))
sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
cv2.imwrite("sheet.png", sheet)
from IPython.display import Image as IPImage, display
display(IPImage("sheet.png"))
print("===RESULT2===")
print(json.dumps({
 "person_frames": sum(len(b) > 0 for b in pboxes), "frames": len(FRAMES),
 "person_box_height_px_median": round(float(np.median(heights))),
 "rfdetr_ms_median_raw": round(float(np.median(t_raw)),1), "rfdetr_ms_median_optimized": round(float(np.median(t_opt)),1) if t_opt else None,
 "pose_people": n_people, "torso_kp_score_median": round(float(np.median(kp_scores)),2),
 "torso_angle_p50_p90_max": np.round(np.percentile(angles,[50,90,100]),1).tolist(), "bent45": int((angles>=45).sum()),
 "gdino_tool_frames": tool_frames, "gdino_labels": dict(label_counter)}, ensure_ascii=False))
print("===END2===")
