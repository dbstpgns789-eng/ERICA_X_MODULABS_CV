# 4차: RTMPose GPU 처리 시간 측정
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","-q","uninstall","-y","onnxruntime"])
subprocess.run([sys.executable,"-m","pip","-q","install","onnxruntime-gpu","rtmlib","huggingface_hub","opencv-python-headless"],check=True)
import onnxruntime as ort
print("providers:", ort.get_available_providers())
import json, time, random, numpy as np, cv2
from huggingface_hub import hf_hub_download
from rtmlib import Body, RTMPose
REPO = "Voxel51/Safe_and_Unsafe_Behaviours"
samples = json.load(open(hf_hub_download(REPO, "samples.json", repo_type="dataset")))
samples = samples["samples"] if isinstance(samples, dict) else samples
random.seed(2)
frames = []
for s in random.sample([x for x in samples if "Intervention" in x["ground_truth"]["label"]], 6):
    cap = cv2.VideoCapture(hf_hub_download(REPO, s["filepath"], repo_type="dataset")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in np.linspace(0, n-1, 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if ok: frames.append(img)
res = {"frames": len(frames)}
for mode in ["lightweight", "balanced"]:
    body = Body(mode=mode, backend="onnxruntime", device="cuda")
    for img in frames[:3]: body(img)  # warmup
    t = []
    for img in frames:
        t0 = time.perf_counter(); body(img); t.append((time.perf_counter()-t0)*1000)
    res[f"body_{mode}_ms_median"] = round(float(np.median(t)), 1)
print("===RESULT4===")
print(json.dumps(res))
print("===END4===")
