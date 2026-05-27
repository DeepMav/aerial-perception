# NanoDet-Plus 1.5x_416 — Aerial 6 Classes

청라 드론 도시감시 영상으로 전이학습된 NanoDet-Plus 모델.

## 모델 정보

| 항목 | 값 |
|---|---|
| 베이스 | nanodet-plus-m-1.5x_416 (COCO 80 pretrained) |
| 백본 | ShuffleNetV2-1.5x |
| Neck | GhostPAN |
| 입력 | 416 × 416 RGB |
| 파라미터 | 7.79M |
| 출력 | 6 클래스 객체 검출 |
| 학습 | 30 epoch, batch 8/GPU × 5 GPU, AdamW |
| 데이터 | AI Hub #183 — 청라 도시감시 10m (Train 15,814 / Val 1,023) |

## 클래스 매핑

| ID | 이름 | 설명 |
|---:|---|---|
| 0 | tree | 가로수·아파트 단지 나무 |
| 1 | structure | 가로등·송전탑·신호등 등 시설물 |
| 2 | building | 아파트·빌딩 |
| 3 | vehicle | 자동차·트럭·버스 |
| 4 | bridge | 다리 |
| 5 | person | 사람 |

## 성능 (Val 1,023장)

| 지표 | 값 |
|---|---:|
| mAP@0.5:0.95 | 0.160 |
| mAP@0.5 (전체) | 30.1% |
| mAP@0.75 | 0.143 |
| vehicle AP@0.5 | 61.0% |
| person AP@0.5 | 41.4% |
| building AP@0.5 | 27.1% |
| structure AP@0.5 | 18.1% |
| tree AP@0.5 | 18.0% |
| bridge AP@0.5 | 14.9% |

## 가중치 다운로드

본 레포에 가중치는 포함되지 않습니다. HuggingFace에서 다운로드:

```bash
huggingface-cli download DeepMav/nanodet-plus-1.5x-aerial-6cls \
    drone_nanodet_416.onnx --local-dir ./weights/
```

또는 Python:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="DeepMav/nanodet-plus-1.5x-aerial-6cls",
    filename="drone_nanodet_416.onnx"
)
```

## 추론 예시

```python
import onnxruntime as ort
import cv2, numpy as np

sess = ort.InferenceSession("weights/drone_nanodet_416.onnx")
img = cv2.imread("aerial.jpg")
img = cv2.resize(img, (416, 416))
x = img.astype(np.float32).transpose(2, 0, 1)[None]
preds = sess.run(None, {"data": x})[0]  # shape [1, 3598, 38]
# postprocessing: 클래스 6개 + bbox 32 (reg_max 8 × 4)
```

후처리 코드는 [`scripts/infer_onnx.py`](../../scripts/infer_onnx.py) 참조 (작성 중).

## 학습 설정

자세한 hyperparameter는 [`config/drone_full.yml`](config/drone_full.yml) 참조.
