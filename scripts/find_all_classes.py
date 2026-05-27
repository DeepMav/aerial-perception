"""6 클래스 모두 등장 영상 시각화
- Train에서 6 클래스 영상 선정 (Val엔 없음)
- Fine-tuned 모델이 가장 많이 잡은 영상 8장
- 학습 영상임을 캡션에 명시
"""
import sys, os
sys.path.insert(0, "/home/kim/drone_nanodet/nanodet")
import torch, cv2, numpy as np
from pathlib import Path
from collections import Counter

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from pycocotools.coco import COCO

DEVICE = "cuda:4"
CLASSES = ["tree","structure","building","vehicle","bridge","person"]
COLORS  = [(34,139,34),(255,165,0),(70,130,180),(220,20,60),(255,215,0),(255,0,255)]
THR = 0.3

TRAIN_JSON = "/home/kim/drone_nanodet/coco_full/train.json"
TRAIN_IMG  = "/home/kim/drone_nanodet/coco_full/train"
OUT = Path("/home/kim/drone_nanodet/viz_all6"); OUT.mkdir(exist_ok=True)

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

def title_bar(img, text, color=(40,100,40)):
    h,w = img.shape[:2]
    bar = np.full((50, w, 3), color, dtype=np.uint8)
    cv2.putText(bar, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return np.vstack([bar, img])

# 1. Train에서 6 클래스 영상 추출
coco = COCO(TRAIN_JSON)
img_ids = coco.getImgIds()

candidates = []
for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    cats = Counter(a["category_id"] for a in anns)
    if len(cats) == 6:
        # 균형 좋은 영상 우선 (각 클래스 최소 카운트가 큰 영상)
        min_cnt = min(cats.values())
        candidates.append((img_id, min_cnt, sum(cats.values()), dict(cats)))

print(f"6 클래스 영상 후보: {len(candidates)}장")
candidates.sort(key=lambda x: (-x[1], -x[2]))  # min count desc

# 2. 모델 로드
ft = P("/home/kim/drone_nanodet/nanodet/config/drone_full.yml",
       "/home/kim/drone_nanodet/workspace/drone_full/model_best/nanodet_model_best.pth")

# 3. 상위 30개 중 모델 검출 가장 많은 것 8장 선정
print("\n=== 추론 결과 ===")
hits = []
for img_id, min_cnt, total, gt_cats in candidates[:30]:
    info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(TRAIN_IMG, info["file_name"])
    preds, _ = ft(img_path)
    detected = Counter()
    for cid, dets in preds.items():
        for d in dets:
            if d[4] >= THR:
                detected[cid] += 1
    ncls = len(detected)
    hits.append((img_id, ncls, dict(detected), gt_cats))
    print(f"  img {img_id}: GT 6cls(min={min_cnt}, total={total}) → 검출 {ncls}cls")

# 6 클래스 검출 우선
hits.sort(key=lambda x: (-x[1], -sum(x[2].values())))
selected = hits[:8]

print("\n=== 최종 선정 ===")
for img_id, ncls, det, gt in selected:
    print(f"  img {img_id}: 검출 {ncls}cls")

# 4. 시각화
for img_id, ncls, det, gt in selected:
    info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(TRAIN_IMG, info["file_name"])
    preds, raw = ft(img_path)
    boxes = []
    for cid, dets in preds.items():
        for d in dets:
            if d[4] >= THR:
                boxes.append([d[0],d[1],d[2],d[3],d[4], cid])
    det_summary = " ".join(f"{CLASSES[c]}:{n}" for c,n in sorted(det.items()))
    panel = title_bar(draw(raw, boxes),
                     f"Fine-tuned ({ncls}/6 cls) - {det_summary}  [Train sample]")
    out_path = OUT / f"all6_id{img_id}.jpg"
    cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  → {out_path}")

print(f"\n저장: {OUT}/*.jpg")
