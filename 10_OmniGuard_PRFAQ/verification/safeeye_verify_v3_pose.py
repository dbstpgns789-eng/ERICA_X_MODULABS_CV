# 3차: 포즈만으로 '설비 개입'과 '통행'을 구별할 수 있는가
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","-q","install","rtmlib","onnxruntime-gpu","huggingface_hub","opencv-python-headless","scikit-learn"],check=True)
import json, time, random, os
import numpy as np, cv2
from huggingface_hub import hf_hub_download
from rtmlib import Body
REPO = "Voxel51/Safe_and_Unsafe_Behaviours"
random.seed(1)
samples = json.load(open(hf_hub_download(REPO, "samples.json", repo_type="dataset")))
samples = samples["samples"] if isinstance(samples, dict) else samples
GROUPS = {"intervention": ["Unauthorized Intervention", "Authorized Intervention"], "walking": ["Safe Walkway", "Safe Walkway Violation"]}
N_PER_CLASS, N_FRAMES = 15, 12
clips = []
for g, labels in GROUPS.items():
    for lab in labels:
        pool = [s for s in samples if s["ground_truth"]["label"] == lab]
        clips += [(g, lab, s["filepath"]) for s in random.sample(pool, N_PER_CLASS)]
try:
    body = Body(mode="balanced", backend="onnxruntime", device="cuda"); dev = "cuda"
except Exception as e:
    print("cuda pose failed, using cpu:", repr(e)[:150]); body = Body(mode="balanced", backend="onnxruntime", device="cpu"); dev = "cpu"
def main_person(kps, sc):
    if len(kps) == 0: return None
    i = int(np.argmax([s.mean() for s in sc])); return kps[i], sc[i]
feats, ys, labs, t_pose = [], [], [], []
for g, lab, fp in clips:
    path = hf_hub_download(REPO, fp, repo_type="dataset")
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seq = []
    for f in np.linspace(0, n - 1, N_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if not ok: continue
        t0 = time.perf_counter(); kps, sc = body(img); t_pose.append((time.perf_counter()-t0)*1000)
        mp = main_person(kps, sc)
        if mp: seq.append(mp)
    cap.release()
    if len(seq) < 3: continue
    K = np.array([k for k, _ in seq]); S = np.array([s for _, s in seq])
    sh, hp = (K[:,5]+K[:,6])/2, (K[:,11]+K[:,12])/2
    torso = np.linalg.norm(sh-hp, axis=1) + 1e-6
    ang = np.degrees(np.arctan2(np.abs((sh-hp)[:,0]), np.abs((sh-hp)[:,1])+1e-6))
    center = (sh+hp)/2
    speed = np.linalg.norm(np.diff(center, axis=0), axis=1) / torso[1:]          # 몸 중심 이동량 (몸통 길이 기준)
    wr = np.stack([K[:,9], K[:,10]], 1); shs = np.stack([K[:,5], K[:,6]], 1)
    reach = (np.linalg.norm(wr - shs, axis=2) / torso[:,None]).max(1)            # 팔 뻗음 정도
    wr_move = np.linalg.norm(np.diff(wr, axis=0), axis=2).mean(1) / torso[1:]     # 손목 움직임
    wr_rel = (wr.mean(1)[:,1] - hp[:,1]) / torso                                  # 손목 높이(엉덩이 기준)
    feats.append([np.mean(speed), np.max(ang), np.mean(ang), np.mean(reach), np.mean(wr_move), np.mean(wr_rel), float(S.mean())])
    ys.append(1 if g == "intervention" else 0); labs.append(lab)
X, y = np.array(feats), np.array(ys)
NAMES = ["body_speed", "torso_angle_max", "torso_angle_mean", "arm_reach", "wrist_motion", "wrist_height", "kp_score"]
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
cv = StratifiedKFold(5, shuffle=True, random_state=0)
res = {}
for name, m, cols in [
    ("logreg_all", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), list(range(6))),
    ("rf_all", RandomForestClassifier(200, random_state=0), list(range(6))),
    ("logreg_speed_only", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), [0]),
    ("logreg_no_speed", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), [1,2,3,4,5]),
]:
    p = cross_val_predict(m, X[:, cols], y, cv=cv)
    res[name] = {"acc": round(accuracy_score(y, p), 3), "f1": round(f1_score(y, p), 3), "cm[[TN,FP],[FN,TP]]": confusion_matrix(y, p).tolist()}
rf = RandomForestClassifier(200, random_state=0).fit(X[:, :6], y)
means = {lab: dict(zip(NAMES, np.round(X[np.array(labs) == lab].mean(0), 2).tolist())) for lab in sorted(set(labs))}
print("===RESULT3===")
print(json.dumps({"pose_device": dev, "pose_ms_median": round(float(np.median(t_pose)), 1),
  "clips_used": int(len(y)), "intervention": int(y.sum()), "walking": int((1-y).sum()), "frames_per_clip": N_FRAMES,
  "cv5": res, "rf_importance": dict(zip(NAMES[:6], np.round(rf.feature_importances_, 3).tolist())),
  "feature_means_by_label": means}, ensure_ascii=False, indent=1))
print("===END3===")
