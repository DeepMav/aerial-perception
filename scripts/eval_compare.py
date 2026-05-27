"""Pretrained(COCO 80) vs Fine-tuned(드론 6) 비교 평가
같은 Val 1023장에 대해 두 모델 추론 → 같은 ground truth로 평가
"""
import json, os, sys, time
from pathlib import Path
import torch
import cv2
import numpy as np

sys.path.insert(0, "/home/kim/drone_nanodet/nanodet")
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from omegaconf import OmegaConf

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEVICE = "cuda:4"
VAL_JSON = "/home/kim/drone_nanodet/coco_full/val.json"
VAL_IMG  = "/home/kim/drone_nanodet/coco_full/val"

# COCO 80 클래스 ID → 우리 6 클래스 ID 매핑
# COCO original: 1=person, 3=car, 4=motorcycle, 6=bus, 8=truck
# Ours: 1=tree, 2=structure, 3=building, 4=vehicle, 5=bridge, 6=person
COCO_TO_OURS = {
    1: 6,   # person → person
    3: 4,   # car → vehicle
    4: 4,   # motorcycle → vehicle
    6: 4,   # bus → vehicle
    8: 4,   # truck → vehicle
}
# tree, structure, building, bridge는 COCO에 매핑 가능한 클래스 없음

class Predictor:
    def __init__(self, cfg, model_path, num_classes_in_ckpt):
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location="cpu")
        load_model_weight(model, ckpt, Logger(-1, use_tensorboard=False))
        self.model = model.to(DEVICE).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.cfg = cfg
        self.num_classes = num_classes_in_ckpt

    @torch.no_grad()
    def predict(self, img_path):
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        meta = dict(img_info={"id":0,"file_name":os.path.basename(img_path),"height":H,"width":W},
                    raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2,0,1)).to(DEVICE)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        results = self.model.inference(meta)
        # results: {img_id: {category_id: [[x,y,x,y,score], ...]}}
        return results[0]


def to_coco_results(results_by_imgid, img_id_map, cat_map=None):
    """추론 결과를 COCO detection result format으로 변환"""
    out = []
    for img_id, per_cls in results_by_imgid.items():
        for cat_id, dets in per_cls.items():
            if cat_map is not None:
                if cat_id not in cat_map: continue
                mapped_cat = cat_map[cat_id]
            else:
                mapped_cat = cat_id + 1  # 0-based → 1-based
            for det in dets:
                x1, y1, x2, y2, score = det
                w, h = x2-x1, y2-y1
                if w <= 0 or h <= 0: continue
                out.append({
                    "image_id": img_id,
                    "category_id": int(mapped_cat),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })
    return out


def evaluate(coco_gt, results, cat_ids_to_eval, label):
    if len(results) == 0:
        print(f"  ⚠️ {label}: 검출 0개")
        return None
    coco_dt = coco_gt.loadRes(results)
    eval = COCOeval(coco_gt, coco_dt, "bbox")
    eval.params.catIds = cat_ids_to_eval
    eval.evaluate(); eval.accumulate(); eval.summarize()
    return {
        "mAP": float(eval.stats[0]),
        "AP_50": float(eval.stats[1]),
        "AP_75": float(eval.stats[2]),
    }


def per_class_ap(coco_gt, results, label):
    if len(results) == 0:
        return {}
    coco_dt = coco_gt.loadRes(results)
    out = {}
    cat_to_name = {c["id"]:c["name"] for c in coco_gt.dataset["categories"]}
    for cat_id, name in cat_to_name.items():
        eval = COCOeval(coco_gt, coco_dt, "bbox")
        eval.params.catIds = [cat_id]
        eval.params.imgIds = coco_gt.getImgIds(catIds=[cat_id])
        if len(eval.params.imgIds) == 0: continue
        eval.evaluate(); eval.accumulate()
        # AP50
        s = eval.eval["precision"]  # [T,R,K,A,M]
        if s.shape[2] == 0: continue
        ap50 = s[0,:,0,0,-1]
        ap_all = s[:,:,0,0,-1]
        ap50_val = ap50[ap50 > -1].mean() if (ap50>-1).any() else -1
        ap_val = ap_all[ap_all > -1].mean() if (ap_all>-1).any() else -1
        out[name] = (float(ap_val), float(ap50_val))
    return out


def run_model(config_path, ckpt_path, label, cat_map=None, n_classes=80):
    print(f"\n{'='*70}\n {label}\n{'='*70}")
    load_config(cfg, config_path)
    pred = Predictor(cfg, ckpt_path, n_classes)
    # 추론
    coco_gt = COCO(VAL_JSON)
    img_ids = coco_gt.getImgIds()
    results_by_imgid = {}
    t0 = time.time()
    for i, img_id in enumerate(img_ids):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG, info["file_name"])
        per_cls = pred.predict(img_path)
        results_by_imgid[img_id] = per_cls
        if (i+1) % 100 == 0:
            print(f"  추론 {i+1}/{len(img_ids)}  ({time.time()-t0:.1f}s)")
    print(f"  추론 완료: {len(img_ids)}장, {time.time()-t0:.1f}s")
    results = to_coco_results(results_by_imgid, None, cat_map)
    print(f"  검출 박스 수: {len(results):,}")
    # 매핑 가능 클래스만 평가
    if cat_map is not None:
        eval_cat_ids = sorted(set(cat_map.values()))
    else:
        eval_cat_ids = sorted(c["id"] for c in coco_gt.dataset["categories"])
    print(f"  평가 클래스 ID: {eval_cat_ids}")
    overall = evaluate(coco_gt, results, eval_cat_ids, label)
    # 클래스별
    per_cls = per_class_ap(coco_gt, results, label)
    return overall, per_cls


# 1. Pretrained (COCO 80 클래스) — base config 사용
pre_overall, pre_per = run_model(
    "/home/kim/drone_nanodet/nanodet/config/nanodet-plus-m-1.5x_416.yml",
    "/home/kim/drone_nanodet/pretrained/nanodet-plus-m-1.5x_416_checkpoint.ckpt",
    "PRETRAINED (COCO 80→매핑→6)",
    cat_map=COCO_TO_OURS, n_classes=80
)

# 2. Fine-tuned (드론 6 클래스)
ft_overall, ft_per = run_model(
    "/home/kim/drone_nanodet/nanodet/config/drone_full.yml",
    "/home/kim/drone_nanodet/workspace/drone_full/model_best/nanodet_model_best.pth",
    "FINE-TUNED (드론 6 클래스)",
    cat_map=None, n_classes=6
)

print("\n" + "="*70)
print(" 비교 요약 — person, vehicle 만 직접 비교 가능 (COCO에 다른 클래스 없음)")
print("="*70)

print(f"\n{'클래스':<12} {'Pretrained AP50':>18} {'Fine-tuned AP50':>18} {'Δ':>10}")
print("-"*65)
for cls in ["person", "vehicle"]:
    pre = pre_per.get(cls, (0,0))
    ft  = ft_per.get(cls, (0,0))
    delta = ft[1] - pre[1]
    print(f"{cls:<12} {pre[1]*100:>16.1f}% {ft[1]*100:>16.1f}% {delta*100:>+9.1f}pp")

print(f"\n[Fine-tuned 전용 클래스 — pretrained 측정 불가]")
for cls in ["tree","structure","building","bridge"]:
    ft = ft_per.get(cls, (0,0))
    print(f"  {cls:<12} {ft[1]*100:.1f}% AP50")

# 저장
out = {
    "pretrained": {"overall": pre_overall, "per_class": {k:list(v) for k,v in pre_per.items()}},
    "fine_tuned": {"overall": ft_overall,  "per_class": {k:list(v) for k,v in ft_per.items()}},
}
with open("/home/kim/drone_nanodet/eval_compare.json","w") as f:
    json.dump(out, f, indent=2)
print("\n저장: /home/kim/drone_nanodet/eval_compare.json")
