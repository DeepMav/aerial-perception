"""GT vs Fine-tuned vs Pretrained 비교 시각화
- Val에서 클래스 다양성 높은 영상 12장 선정
- 한 그림에 3패널: GT / Fine-tuned 예측 / Pretrained 예측
- 4K를 1280으로 리사이즈
- 클래스별 색상
"""
import sys, os, json, random
import torch, cv2
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/home/kim/drone_nanodet/nanodet")
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

from pycocotools.coco import COCO

random.seed(7)

DEVICE = "cuda:4"
VAL_JSON = "/home/kim/drone_nanodet/coco_full/val.json"
VAL_IMG  = "/home/kim/drone_nanodet/coco_full/val"
OUT_DIR  = Path("/home/kim/drone_nanodet/viz")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["tree","structure","building","vehicle","bridge","person"]
COLORS  = [(34,139,34),(255,165,0),(70,130,180),(220,20,60),(255,215,0),(255,0,255)]  # 클래스별 BGR

# Pretrained → 우리 매핑
COCO_TO_OURS = {1: 6, 3: 4, 4: 4, 6: 4, 8: 4}  # 1-indexed in COCO eval, 0-indexed in NanoDet predict

class Predictor:
    def __init__(self, config_path, model_path):
        load_config(cfg, config_path)
        self.cfg = type('C', (), {})()
        self.cfg.data = cfg.data
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location="cpu")
        load_model_weight(model, ckpt, Logger(-1, use_tensorboard=False))
        self.model = model.to(DEVICE).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.input_size = cfg.data.val.input_size

    @torch.no_grad()
    def predict(self, img_path):
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        meta = dict(img_info={"id":0,"file_name":os.path.basename(img_path),"height":H,"width":W},
                    raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2,0,1)).to(DEVICE)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        results = self.model.inference(meta)
        return results[0], img  # {cat: [[x,y,x,y,score],...]}, original BGR

def draw_boxes(img, boxes_with_class, score_thr=0.3, label_classes=CLASSES, downscale=None):
    """boxes_with_class: list of (x1,y1,x2,y2,score,class_id)"""
    img = img.copy()
    if downscale:
        h, w = img.shape[:2]
        new_w = downscale
        scale = new_w / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    else:
        scale = 1.0
    for x1,y1,x2,y2,score,cid in boxes_with_class:
        if score < score_thr: continue
        x1,y1,x2,y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
        color = COLORS[cid % len(COLORS)]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f"{label_classes[cid]}:{score:.2f}" if score < 1.0 else label_classes[cid]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-5), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

def add_title(img, text):
    h, w = img.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return np.vstack([bar, img])

# === 다양한 클래스 포함된 12장 선정 ===
print("Loading GT ...")
coco_gt = COCO(VAL_JSON)
img_ids = coco_gt.getImgIds()

# 각 이미지에 등장 클래스 수 + person/bridge 우선
img_scores = []
for img_id in img_ids:
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    cats = set(a["category_id"] for a in anns)
    score = len(cats)  # 다양성
    if 6 in cats: score += 2  # person 있으면 우선
    if 5 in cats: score += 3  # bridge 있으면 더 우선
    img_scores.append((img_id, score, len(anns)))
img_scores.sort(key=lambda x: (-x[1], -x[2]))
selected_ids = [x[0] for x in img_scores[:50]]
random.shuffle(selected_ids)
selected_ids = selected_ids[:12]
print(f"선정된 영상 ID: {selected_ids}")

# === 모델 2개 ===
print("Loading Pretrained ...")
pre = Predictor(
    "/home/kim/drone_nanodet/nanodet/config/nanodet-plus-m-1.5x_416.yml",
    "/home/kim/drone_nanodet/pretrained/nanodet-plus-m-1.5x_416_checkpoint.ckpt"
)
print("Loading Fine-tuned ...")
ft = Predictor(
    "/home/kim/drone_nanodet/nanodet/config/drone_full.yml",
    "/home/kim/drone_nanodet/workspace/drone_full/model_best/nanodet_model_best.pth"
)

DOWNSCALE = 960  # 4K → 960 가로

# === 시각화 ===
for i, img_id in enumerate(selected_ids):
    info = coco_gt.loadImgs(img_id)[0]
    img_path = os.path.join(VAL_IMG, info["file_name"])
    print(f"[{i+1}/12] {info['file_name']}")

    # 추론
    pre_preds, raw = pre.predict(img_path)
    ft_preds, _    = ft.predict(img_path)

    # GT 박스
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    gt_boxes = []
    for a in anns:
        x,y,w,h = a["bbox"]
        cid = a["category_id"] - 1  # 1-indexed → 0-indexed
        gt_boxes.append([x,y,x+w,y+h,1.0,cid])

    # Pretrained → 우리 매핑
    pre_boxes = []
    for cat_id, dets in pre_preds.items():
        # NanoDet predict returns 0-indexed cat_id
        coco_id_1based = cat_id + 1
        if coco_id_1based not in COCO_TO_OURS: continue
        our_cls = COCO_TO_OURS[coco_id_1based] - 1
        for d in dets:
            x1,y1,x2,y2,s = d
            pre_boxes.append([x1,y1,x2,y2,s,our_cls])

    # Fine-tuned (0-indexed → 그대로)
    ft_boxes = []
    for cat_id, dets in ft_preds.items():
        for d in dets:
            x1,y1,x2,y2,s = d
            ft_boxes.append([x1,y1,x2,y2,s,cat_id])

    panel_gt  = add_title(draw_boxes(raw, gt_boxes,  score_thr=0.0, downscale=DOWNSCALE),
                          f"[1] Ground Truth  ({len(gt_boxes)} objects)")
    panel_ft  = add_title(draw_boxes(raw, ft_boxes,  score_thr=0.3, downscale=DOWNSCALE),
                          f"[2] Fine-tuned (drone 6 classes, mAP@0.5={0.301:.2f})")
    panel_pre = add_title(draw_boxes(raw, pre_boxes, score_thr=0.3, downscale=DOWNSCALE),
                          f"[3] Pretrained (COCO 80 -> person/vehicle only)")

    combined = np.vstack([panel_gt, panel_ft, panel_pre])
    out_path = OUT_DIR / f"compare_{i+1:02d}_id{img_id}.jpg"
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  → {out_path}")

print(f"\n완료: {OUT_DIR}/compare_*.jpg")
