"""본 학습용 클래스 인지 샘플링 + COCO 변환
- person/bridge 포함 프레임 → ALL
- vehicle 포함 프레임 → 1/15
- 그 외 → 1/60
- Train: train/ 디렉토리 (재구성된 198 클립)
- Val:   val/   디렉토리 (재구성된 50 클립)
"""
import json, glob, os, random, re
from pathlib import Path
from collections import Counter, defaultdict
random.seed(42)

LABEL_TRAIN = "/tmp/drone_183_labels/train"
LABEL_VAL   = "/tmp/drone_183_labels/val"
IMG_TRAIN   = "/home/kim/drone_nanodet/images_train"
IMG_VAL     = "/home/kim/drone_nanodet/images_val"
OUT_DIR     = Path("/home/kim/drone_nanodet/coco_full")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["tree","structure","building","vehicle","bridge","person"]
CAT_ID  = {c:i+1 for i,c in enumerate(CLASSES)}

def stem_clip(p):
    return os.path.basename(p).rsplit('_',1)[0]

def class_aware_select(json_files):
    """클래스 인지 샘플링"""
    clip_frames = defaultdict(list)
    for f in json_files:
        clip_frames[stem_clip(f)].append(f)

    selected = []
    stats = Counter()
    for clip, frames in clip_frames.items():
        frames = sorted(frames)
        for i, f in enumerate(frames):
            try:
                with open(f) as fp: d = json.load(fp)
            except: continue
            labels = {a.get("label") for a in d.get("annotations",[])}
            if "person" in labels or "bridge" in labels:
                selected.append(f); stats["person/bridge ALL"] += 1
            elif "vehicle" in labels:
                if i % 15 == 0:
                    selected.append(f); stats["vehicle 1/15"] += 1
            else:
                if i % 60 == 0:
                    selected.append(f); stats["기타 1/60"] += 1
    return selected, stats

def to_coco(json_files, img_root, split):
    print(f"\n=== {split} ===")
    print(f"  입력 JSON: {len(json_files):,}")
    images, annotations = [], []
    ann_id = 1
    cls_cnt = Counter()
    img_dir = OUT_DIR / split
    img_dir.mkdir(exist_ok=True)
    missing_img = 0

    for img_id, jp in enumerate(json_files, 1):
        try:
            with open(jp) as f: d = json.load(f)
        except: continue
        fn = d.get("filename","")
        pp = d.get("parent_path","").strip("/")
        ip = Path(img_root) / pp / fn
        if not ip.exists():
            missing_img += 1
            continue
        anns = d.get("annotations",[])
        if not anns: continue
        W = d["metadata"]["width"]; H = d["metadata"]["height"]

        # 평탄화 파일명 — clip stem 사용해 충돌 회피
        flat = fn   # 파일명이 고유 (날짜·시간·시퀀스·프레임번호 포함)
        dst = img_dir / flat
        if not dst.exists():
            try: dst.symlink_to(ip)
            except FileExistsError: pass

        valid_anns = 0
        for a in anns:
            lbl = a.get("label")
            if lbl not in CAT_ID: continue
            pts = a.get("points",[])
            if len(pts) < 4: continue
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
            w=x2-x1; h=y2-y1
            if w<=0 or h<=0: continue
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": CAT_ID[lbl],
                "bbox": [x1,y1,w,h], "area": w*h, "iscrowd":0, "segmentation":[]
            })
            ann_id += 1
            cls_cnt[lbl] += 1
            valid_anns += 1
        if valid_anns > 0:
            images.append({"id":img_id,"file_name":flat,"width":W,"height":H})

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id":i+1,"name":c,"supercategory":"drone"} for i,c in enumerate(CLASSES)]
    }
    out = OUT_DIR / f"{split}.json"
    with open(out,"w") as f: json.dump(coco, f)
    print(f"  영상 누락: {missing_img}")
    print(f"  → images={len(images):,}  ann={len(annotations):,}")
    print(f"  클래스별 ann:")
    for c in CLASSES:
        n = cls_cnt[c]
        print(f"    {c:<10s} {n:>7,}")
    return out

# Train
train_json = sorted(glob.glob(f"{LABEL_TRAIN}/**/*.json", recursive=True))
print(f"Train JSON 전체: {len(train_json):,}")
train_sel, train_stats = class_aware_select(train_json)
print(f"Train 샘플링 결과: {len(train_sel):,}")
for k,v in train_stats.items(): print(f"  {k}: {v:,}")

# Val
val_json = sorted(glob.glob(f"{LABEL_VAL}/**/*.json", recursive=True))
print(f"\nVal JSON 전체: {len(val_json):,}")
val_sel, val_stats = class_aware_select(val_json)
print(f"Val 샘플링 결과: {len(val_sel):,}")
for k,v in val_stats.items(): print(f"  {k}: {v:,}")

to_coco(train_sel, IMG_TRAIN, "train")
to_coco(val_sel,   IMG_VAL,   "val")
print(f"\n출력: {OUT_DIR}")
