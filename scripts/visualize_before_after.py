"""Before(Pretrained 학습 전) vs After(Fine-tuned 학습 후) 2패널 비교
- 동일 영상 위아래 2패널
- 위: 학습 전 COCO 모델 (person/vehicle만 인식 가능)
- 아래: 30 epoch 전이학습 후 모델 (6 클래스 모두)
"""
import sys, os, re, json
sys.path.insert(0, "/home/kim/drone_nanodet/nanodet")
import torch, cv2, numpy as np
from pathlib import Path

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from pycocotools.coco import COCO

DEVICE = "cuda:4"
CLASSES = ["tree","structure","building","vehicle","bridge","person"]
COLORS  = [(34,139,34),(255,165,0),(70,130,180),(220,20,60),(255,215,0),(255,0,255)]
COCO_TO_OURS_0 = {0:5, 2:3, 3:3, 5:3, 7:3}

class P:
    def __init__(self, cfg_path, model_path):
        load_config(cfg, cfg_path)
        m = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location="cpu")
        load_model_weight(m, ckpt, Logger(-1, use_tensorboard=False))
        self.m = m.to(DEVICE).eval()
        self.pipe = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.isz = cfg.data.val.input_size
    @torch.no_grad()
    def __call__(self, p):
        img = cv2.imread(p); H,W = img.shape[:2]
        meta = dict(img_info={"id":0,"file_name":"","height":H,"width":W}, raw_img=img, img=img)
        meta = self.pipe(None, meta, self.isz)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2,0,1)).to(DEVICE)
        meta = naive_collate([meta]); meta["img"] = stack_batch_img(meta["img"], divisible=32)
        return self.m.inference(meta)[0], img

def draw(img, boxes, downscale=1280):
    h,w = img.shape[:2]; scale = downscale/w
    img = cv2.resize(img, (int(w*scale), int(h*scale))).copy()
    for x1,y1,x2,y2,score,cid in boxes:
        x1,y1,x2,y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
        color = COLORS[cid]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f"{CLASSES[cid]}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-5), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

def title_bar(img, text, color=(40,40,40)):
    h,w = img.shape[:2]
    bar = np.full((50, w, 3), color, dtype=np.uint8)
    cv2.putText(bar, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    return np.vstack([bar, img])

VAL_JSON = "/home/kim/drone_nanodet/coco_full/val.json"
VAL_IMG  = "/home/kim/drone_nanodet/coco_full/val"
OUT = Path("/home/kim/drone_nanodet/viz_before_after"); OUT.mkdir(exist_ok=True)
THR = 0.3

coco = COCO(VAL_JSON)
pre = P("/home/kim/drone_nanodet/nanodet/config/nanodet-plus-m-1.5x_416.yml",
        "/home/kim/drone_nanodet/pretrained/nanodet-plus-m-1.5x_416_checkpoint.ckpt")
ft  = P("/home/kim/drone_nanodet/nanodet/config/drone_full.yml",
        "/home/kim/drone_nanodet/workspace/drone_full/model_best/nanodet_model_best.pth")

# 차이 가장 큰 영상 6장 선정: vehicle/person 잘 보이는 것 우선
target_ids = [9, 830, 833, 839, 877, 12]   # viz 통계에서 비교 풍부했던 것들

for img_id in target_ids:
    info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(VAL_IMG, info["file_name"])
    print(f"img {img_id}: {info['file_name']}")
    pre_preds, raw = pre(img_path)
    ft_preds, _    = ft(img_path)

    # Pretrained → 우리 매핑
    pre_boxes = []
    for cid, dets in pre_preds.items():
        if cid not in COCO_TO_OURS_0: continue
        for d in dets:
            if d[4] >= THR:
                pre_boxes.append([d[0],d[1],d[2],d[3],d[4], COCO_TO_OURS_0[cid]])
    ft_boxes = []
    for cid, dets in ft_preds.items():
        for d in dets:
            if d[4] >= THR:
                ft_boxes.append([d[0],d[1],d[2],d[3],d[4], cid])

    n_pre = len(pre_boxes)
    n_ft = len(ft_boxes)
    panel_pre = title_bar(draw(raw, pre_boxes), f"BEFORE: COCO Pretrained  -  {n_pre} detections", color=(50,50,150))
    panel_ft  = title_bar(draw(raw, ft_boxes),  f"AFTER : Fine-tuned 30 epoch  -  {n_ft} detections", color=(50,120,50))
    combined = np.vstack([panel_pre, panel_ft])
    out_path = OUT / f"before_after_id{img_id}.jpg"
    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  → {out_path}  (Pre {n_pre} -> FT {n_ft})")

print(f"\n저장: {OUT}/*.jpg")
