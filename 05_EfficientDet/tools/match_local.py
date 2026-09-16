"""정답 vs 예측을 IoU로 대조해서 점수가 어디서 새는지 센다.

Colab이 아니라 로컬 CPU에서 돌린 분석 스크립트. 아래 경로 네 개를 환경에 맞게 바꿔서 쓴다.
"""
import os

YAEDP_DIR   = r'path/to/Yet-Another-EfficientDet-Pytorch'   # git clone 받은 저장소
CKPT_PATH   = r'../weights/efficientdet-d0_87_704.pth'       # 평가할 가중치
YML_PATH    = r'../weights/my_car_detect_proj.yml'           # 프로젝트 설정
ARCHIVE_ZIP = r'path/to/archive.zip'                         # 데이터셋 zip
OUT_DIR     = r'.'                                           # 결과 저장 위치

import sys, os, json, zipfile, io
sys.path.insert(0, YAEDP_DIR)
os.chdir(YAEDP_DIR)
import torch, yaml, numpy as np
from collections import Counter
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

CKPT = CKPT_PATH
YML  = YML_PATH
params = yaml.safe_load(open(YML))
model = EfficientDetBackbone(compound_coef=0, num_classes=len(params['obj_list']),
                             ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
model.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False)); model.eval()

z = zipfile.ZipFile(ARCHIVE_ZIP)
base = 'Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/test/'
gt = json.loads(z.read(base + '_annotations.coco.json'))
cats = {c['id']: c['name'] for c in gt['categories']}
id2file = {i['id']: i['file_name'] for i in gt['images']}
gt_by_file = {}
for a in gt['annotations']:
    x, y, w, h = a['bbox']; gt_by_file.setdefault(id2file[a['image_id']], []).append(([x, y, x + w, y + h], a['category_id']))
tmp = os.path.join(OUT_DIR, 'effdet_test_imgs')
files = sorted(gt_by_file)[:6]

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0

def predict(path, thr, nms):
    ori, framed, metas = preprocess(path, max_size=512)
    x = torch.stack([torch.from_numpy(f) for f in framed], 0).float().permute(0, 3, 1, 2)
    with torch.no_grad():
        feats, reg, cls, anchors = model(x)
        out = postprocess(x, anchors, reg, cls, BBoxTransform(), ClipBoxes(), thr, nms)
    return invert_affine(metas, out)[0]

out = io.StringIO()
for thr, nms, tag in [(0.4, 0.2, '노트북 그림 설정 (thr 0.4, nms 0.2)'), (0.05, 0.5, 'coco_eval 설정 (thr 0.05, nms 0.5) → 상위 100개')]:
    out.write(f'\n===== {tag} =====\n')
    tot = Counter()
    for f in files:
        p = predict(os.path.join(tmp, f), thr, nms)
        boxes, scores, labels = p['rois'], p['scores'], p['class_ids']
        order = np.argsort(-scores)[:100]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        gts = gt_by_file[f]
        used = set(); tp = 0; tp_cls = 0; dup = 0; fp_bg = 0; loc_bad = 0
        for b, s, l in zip(boxes, scores, labels):
            pred_cid = int(l) + 1                      # coco_eval과 같은 규칙
            best_i, best_iou = -1, 0
            for gi, (gb, gc) in enumerate(gts):
                v = iou(b, gb)
                if v > best_iou: best_iou, best_i = v, gi
            if best_iou >= 0.5:
                if best_i in used: dup += 1
                else:
                    used.add(best_i); tp += 1
                    if gts[best_i][1] == pred_cid: tp_cls += 1
            elif best_iou >= 0.2: loc_bad += 1
            else: fp_bg += 1
        n_pred = len(boxes); n_gt = len(gts)
        tot.update({'예측': n_pred, '정답': n_gt, '맞음(IoU≥.5)': tp, '맞음+클래스도맞음': tp_cls, '중복': dup, '위치어긋남(.2~.5)': loc_bad, '배경오탐(<.2)': fp_bg, '놓친정답': n_gt - tp})
        out.write(f'{f[:14]} 정답{n_gt:3d} 예측{n_pred:3d} | 맞음{tp:3d} (클래스까지 {tp_cls:3d}) 중복{dup:3d} 위치어긋{loc_bad:3d} 배경{fp_bg:3d} | 놓침{n_gt-tp:3d}\n')
    out.write('합계: ' + '  '.join(f'{k}={v}' for k, v in tot.items()) + '\n')
    if tot['예측']:
        out.write(f'정밀도(클래스 포함) = {tot["맞음+클래스도맞음"]}/{tot["예측"]} = {tot["맞음+클래스도맞음"]/tot["예측"]:.2f}   재현율 = {tot["맞음+클래스도맞음"]}/{tot["정답"]} = {tot["맞음+클래스도맞음"]/tot["정답"]:.2f}\n')

# 클래스 혼동: 맞은 것들의 예측 클래스 vs 정답 클래스
out.write('\n===== 위치는 맞은 박스의 클래스 대조 (thr 0.3) =====\n')
conf = Counter()
for f in files:
    p = predict(os.path.join(tmp, f), 0.3, 0.5)
    for b, l in zip(p['rois'], p['class_ids']):
        for gb, gc in gt_by_file[f]:
            if iou(b, gb) >= 0.5:
                conf[(cats[gc], cats.get(int(l) + 1, f'label{int(l)}'))] += 1; break
for (g, p_), n in conf.most_common(12):
    out.write(f'  정답 {g:10s} → 예측 {p_:10s} : {n}\n')

io.open(os.path.join(OUT_DIR, 'effdet_match.txt'), 'w', encoding='utf-8').write(out.getvalue())
print('done')
