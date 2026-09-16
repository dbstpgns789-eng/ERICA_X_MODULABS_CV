"""학습된 EfficientDet-D0로 test 이미지 6장 추론, 정답(초록) vs 예측(노랑) 그림 저장.

Colab이 아니라 로컬 CPU에서 돌린 분석 스크립트. 아래 경로 네 개를 환경에 맞게 바꿔서 쓴다.
"""
import os

YAEDP_DIR   = r'path/to/Yet-Another-EfficientDet-Pytorch'   # git clone 받은 저장소
CKPT_PATH   = r'../weights/efficientdet-d0_87_704.pth'       # 평가할 가중치
YML_PATH    = r'../weights/my_car_detect_proj.yml'           # 프로젝트 설정
ARCHIVE_ZIP = r'path/to/archive.zip'                         # 데이터셋 zip
OUT_DIR     = r'.'                                           # 결과 저장 위치

import sys, os, json, glob, zipfile, io
sys.path.insert(0, YAEDP_DIR)
os.chdir(YAEDP_DIR)

import torch, yaml, cv2, numpy as np
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

CKPT = CKPT_PATH
YML  = YML_PATH
OUT  = OUT_DIR

params = yaml.safe_load(open(YML))
obj_list = params['obj_list']
model = EfficientDetBackbone(compound_coef=0, num_classes=len(obj_list),
                             ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
model.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False))
model.requires_grad_(False); model.eval()

# 데이터: archive.zip에서 test 이미지와 라벨을 임시로 풀기
z = zipfile.ZipFile(ARCHIVE_ZIP)
base = 'Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/test/'
gt = json.loads(z.read(base + '_annotations.coco.json'))
id2file = {i['id']: i['file_name'] for i in gt['images']}
gt_by_file = {}
for a in gt['annotations']:
    gt_by_file.setdefault(id2file[a['image_id']], []).append((a['bbox'], a['category_id']))
cats = {c['id']: c['name'] for c in gt['categories']}
tmpdir = os.path.join(OUT, 'effdet_test_imgs'); os.makedirs(tmpdir, exist_ok=True)
files = sorted(gt_by_file.keys())[:6]
for f in files:
    open(os.path.join(tmpdir, f), 'wb').write(z.read(base + f))

def predict(path, threshold=0.05, iou_threshold=0.5):
    ori, framed, metas = preprocess(path, max_size=512)
    x = torch.stack([torch.from_numpy(f) for f in framed], 0).float().permute(0, 3, 1, 2)
    with torch.no_grad():
        feats, reg, cls, anchors = model(x)
        out = postprocess(x, anchors, reg, cls, BBoxTransform(), ClipBoxes(), threshold, iou_threshold)
    return ori[0], invert_affine(metas, out)[0]

summary = []
tiles = []
for f in files:
    img, pred = predict(os.path.join(tmpdir, f))
    img = img.copy()
    gts = gt_by_file[f]
    for (x, y, w, h), cid in gts:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
    n_hi = int((pred['scores'] > 0.3).sum()) if len(pred['scores']) else 0
    for j in range(len(pred['rois'])):
        x1, y1, x2, y2 = pred['rois'][j].astype(int)
        s = float(pred['scores'][j])
        color = (255, 255, 0)
        if s <= 0.3: continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    summary.append((f[:22], len(gts), len(pred['rois']), n_hi,
                    float(pred['scores'].max()) if len(pred['scores']) else 0.0,
                    [round(float(s), 2) for s in sorted(pred['scores'], reverse=True)[:5]]))
    tiles.append(cv2.resize(img, (480, 480)))

grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
cv2.imwrite(os.path.join(OUT, 'effdet_pred_grid_hi.png'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

with io.open(os.path.join(OUT, 'effdet_infer_summary.txt'), 'w', encoding='utf-8') as fo:
    fo.write('이미지 | 정답 박스 | 예측 박스(≥0.05) | 예측(≥0.3) | 최고 점수 | 상위 점수 5개\n')
    for r in summary:
        fo.write(f'{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]}\n')
    # 클래스 분포
    from collections import Counter
    fo.write('\n정답 클래스 분포(6장): ' + str(Counter(cats[c] for f in files for _, c in gt_by_file[f])) + '\n')
    fo.write('이미지 크기: ' + str(cv2.imread(os.path.join(tmpdir, files[0])).shape) + '\n')
print('done')
