# aerial-perception

> 마이크로 항공기(MAV)를 위한 오픈소스 비전 인식 모델.
> NanoDet-Plus를 AI Hub 드론 항공 영상으로 전이학습한 6 클래스 객체 인식 모델 — ONNX·안드로이드 배포 지원.

<p align="center">
  <img src="viz/before_after_id830.jpg" width="900" alt="학습 전 vs 학습 후"/>
  <br/>
  <em>위: COCO 사전학습 모델 — 항공 영상에서 거의 인식 못 함.
  아래: 30 epoch 전이학습 후 — 다리·차량·시설물 모두 검출.</em>
</p>

## 프로젝트 상태

- v0.1 — 실험적 베이스라인 (mAP@0.5 = 30.1%)
- 상용 배포용 아님. 프로토타입·연구·교육용으로 적합.
- 향후 계획: 멀티 모델 동물원 + LoRA 핫스왑 (아래 로드맵 참조).

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 도메인 | 드론 항공 영상 객체 인식 |
| 베이스 모델 | NanoDet-Plus-m-1.5x_416 (파라미터 7.79M) |
| 클래스 | `tree`, `structure`, `building`, `vehicle`, `bridge`, `person` |
| 학습 데이터 | AI Hub #183 — 청라 도시감시 10m 드론 영상 |
| Train / Val | 15,814 / 1,023 프레임 (4K JPG) |
| 산출물 | PyTorch (`.pth`, 31MB) · ONNX (11MB) |
| 라이선스 | Apache 2.0 |

모델 가중치는 본 레포에 포함되지 않으며 HuggingFace에서 다운로드:
[`harveykim/nanodet-plus-1.5x-aerial-6cls`](https://huggingface.co/harveykim/nanodet-plus-1.5x-aerial-6cls)

---

## 결과 — 사전학습 vs 전이학습

### 클래스별 AP@0.5 (Val 1,023장 기준)

| 클래스 | 사전학습 (COCO 80) | 전이학습 (Drone 6) | 향상 |
|---|---:|---:|---:|
| vehicle | 4.2% | 61.0% | +56.8pp |
| person | 1.3% | 41.4% | +40.1pp |
| building | — | 27.1% | 신규 학습 |
| structure | — | 18.1% | 신규 학습 |
| tree | — | 18.0% | 신규 학습 |
| bridge | — | 14.9% | 신규 학습 |
| 전체 mAP@0.5 | — | 30.1% | |

COCO 사전학습 모델은 tree·structure·bridge 같은 클래스 자체가 없음.
전이학습을 통해 0% → 14–27% 신규 학습. vehicle·person은 +40–57pp 폭발적 향상.

---

## 시각적 비교

### 6 클래스 모두 검출하는 데모

<p align="center">
  <img src="viz/all6_id9999.jpg" width="900" alt="6 클래스 모두 검출"/>
  <br/>
  <em>한 프레임에서 6개 클래스 모두 검출 (training sample).
  도시 도로 장면에서 나무·시설물·건물·차량·다리·사람 동시 인식.</em>
</p>

### 학습 전 vs 학습 후

<p align="center">
  <img src="viz/before_after_id877.jpg" width="900" alt="사람·시설물 검출"/>
  <br/>
  <em>위: 사전학습 모델 — 좁쌀만한 사람을 거의 못 잡음.
  아래: 전이학습 모델 — 작은 사람과 다수의 시설물 새로 검출.</em>
</p>

### 3 패널 종합 비교 — Ground Truth / 전이학습 / 사전학습

<p align="center">
  <img src="viz/compare_03_id880.jpg" width="900" alt="3패널 비교"/>
  <br/>
  <em>[1] Ground Truth (사람이 단 정답) · [2] 전이학습 모델 · [3] 사전학습 (사실상 백지).</em>
</p>

### 박스 색상 범례

| 클래스 | 색상 |
|---|---|
| tree | 녹색 (forest green) |
| structure | 주황 (orange) |
| building | 청색 (steel blue) |
| vehicle | 적색 (crimson) |
| bridge | 황색 (gold) |
| person | 분홍 (magenta) |

---

## 빠른 시작

### 1. 환경 설치

```bash
git clone git@github.com:DeepMav/aerial-perception.git
cd aerial-perception
pip install -r requirements.txt
```

### 2. 모델 가중치 다운로드 (HuggingFace)

```bash
# huggingface-cli 사용
pip install huggingface_hub
huggingface-cli download harveykim/nanodet-plus-1.5x-aerial-6cls \
    drone_nanodet_416.onnx --local-dir ./weights/
```

### 3. ONNX 추론 (예정)

```bash
python scripts/infer_onnx.py \
    --model ./weights/drone_nanodet_416.onnx \
    --image path/to/aerial.jpg \
    --score-threshold 0.3
```

### 안드로이드 배포

1. ONNX → NCNN (`.param` + `.bin`) 변환
2. NanoDet 공식 [Android NCNN 데모](https://github.com/RangiLyu/nanodet/tree/main/demo_android_ncnn) 활용
3. 클래스명을 본 프로젝트의 6 클래스로 교체
4. 예상 추론 속도: 8–25ms (Snapdragon 8 Gen 1+)

---

## 학습 재현

학습을 직접 재현하려면 AI Hub에서 데이터셋 #183을 다운로드 후
경로를 본인 환경에 맞게 수정 필요:

```bash
# 1. AI Hub 데이터셋 다운로드 (별도 신청 후 aihubshell)
#    https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=183

# 2. 라벨 zip 압축 해제, JSON → COCO 변환
python scripts/make_coco_full.py
# (스크립트 상단 경로 변수를 본인 환경에 맞게 수정)

# 3. NanoDet 학습 (NanoDet 별도 설치 필요)
python /path/to/nanodet/tools/train.py models/nanodet_plus_1.5x_416/config/drone_full.yml

# 4. 평가·시각화
python scripts/eval_compare.py
python scripts/visualize.py
```

`scripts/` 내 모든 파일의 경로 변수는 본인 환경에 맞게 수정해야 합니다.

---

## 로드맵

```
v0.1 (현재)    NanoDet-Plus 베이스라인 (mAP 30%)
v0.2           레포 정리, 모델 카드, HuggingFace 공개
v0.3           YOLOv8 / RT-DETR 베이스라인 추가 + 자동 리더보드
v0.4           LoRA 핫스왑 엔진 구현
                 - 도메인 어댑터: 도시 / 농촌 / 배송 / 산악
                 - 런타임 모델 전환 (1초 미만)
v0.5           통합 Python SDK + CLI
v0.6           모바일 배포 (Android NCNN/TFLite, Jetson)
v1.0           정식 릴리즈, 논문, HuggingFace Spaces 데모
```

### LoRA 핫스왑이 왜 필요한가

도메인마다 완전 학습 모델을 따로 만들면 약 30MB × N이 필요합니다.
LoRA 어댑터를 사용하면:

- 2–5MB 어댑터 per 도메인 (cheongna-urban, dusting-rural, delivery, slam-mountain 등)
- 1초 미만에 런타임 교체
- 베이스 모델은 PyTorch / ONNX 하나만 로드

드론 한 대로 도시·농촌·산악 등 다양한 환경 대응 가능.

---

## 한계 및 주의사항

- 단일 지역 데이터 — 모든 학습 영상이 청라(인천) 지역. 다른 도시 일반화 미검증.
- 30 epoch — Epoch 15부터 성능 plateau. 더 긴 학습으로 약 40% 도달 가능.
- 입력 해상도 416×416 — 4K 원본을 416으로 줄이면서 작은 객체(사람) 정보 손실.
  실서비스에는 SAHI(슬라이싱 추론) 권장.
- Validation 불균형 — `bridge` 클래스는 Val에 135개뿐.
- 클래스 불균형 — `tree`가 전체 어노테이션의 54%. 학습 시 클래스 인지 샘플링 적용.

---

## 프로젝트 구조

```
aerial-perception/
├── models/
│   └── nanodet_plus_1.5x_416/
│       ├── config/drone_full.yml         # 학습 설정
│       └── README.md                      # 모델 카드 (HuggingFace 링크)
├── scripts/                                # 데이터 변환, 학습, 평가
│   ├── make_coco_full.py
│   ├── eval_compare.py
│   ├── visualize.py
│   ├── visualize_before_after.py
│   └── find_all_classes.py
├── viz/                                    # 시각화 결과
└── docs/                                   # 추가 문서
```

---

## 감사의 글

- 베이스 모델: [NanoDet-Plus](https://github.com/RangiLyu/nanodet) — [@RangiLyu](https://github.com/RangiLyu) (Apache 2.0)
- 데이터셋: [AI Hub #183 드론 이동체 인지 영상](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=183)
  연구 목적 사용. 원본 데이터는 본 레포에 재배포되지 않음.
- 조직: [DeepMav](https://github.com/DeepMav)

## 라이선스

본 레포는 별도 라이선스를 부여하지 않습니다 (All Rights Reserved).
연구·교육·상용 활용을 원하시면 별도 협의 바랍니다.

다음은 별도 라이선스 적용:

- **NanoDet 베이스 코드** (NanoDet 원본 구조 등):
  원작자 [@RangiLyu](https://github.com/RangiLyu)의 Apache 2.0 라이선스 유지.
- **모델 가중치 (HuggingFace)**:
  [`harveykim/nanodet-plus-1.5x-aerial-6cls`](https://huggingface.co/harveykim/nanodet-plus-1.5x-aerial-6cls)는
  **CC BY-NC 4.0** 라이선스 (비상용·출처 표시).

상용 사용·라이선스 협의: GitHub Issues 또는 조직 [DeepMav](https://github.com/DeepMav) 문의.
