# Flood_Water_system

## 모델 학습 코드 : [모델 생성(학습) 코드](https://github.com/kimyoungjin2023/flood_Water_Level_Detection)

파일 구조

```

flood_system/
├── main.py                          ← 실행 진입점 (10단계 파이프라인)
├── utils/
│   ├── config.py                    ← 전체 설정 (새 모델명, 5cls 차량표)
│   ├── dataclasses.py               ← 공통 데이터 클래스
│   └── image_utils.py               ← 전처리 유틸
├── models/
│   └── loader.py                    ← 전체 모델 로딩 (ResNet18 + EfficientNet-B0 ×3 + YOLO ×5 + Keras)
├── geometry/                        ← [신규] 기하학 독립 모듈
│   ├── ground_plane.py              ← 소실점 검출 + Homography
│   └── waterline.py                 ← RANSAC 수면선 추출
├── confidence/                      ← [신규] 신뢰도 독립 모듈
│   └── calibration.py               ← Dynamic Confidence (bbox크기/경계/pose visibility/자세)
├── estimators/
│   ├── person.py                    ← 사람 (scale calibration + 자세판별)
│   ├── car.py                       ← 차량 (EfficientNet 5cls + orientation)
│   ├── infrastructure.py            ← 표지판/시선유도봉
│   └── flood_seg.py                 ← Seg 수위 추정
├── fusion/
│   └── weighted_fusion.py           ← IQR + Uncertainty-aware fusion
└── visualization/
    └── visualizer.py                ← 6패널 시각화 + 불확실도 오차막대

```

현재 DEMO 모델- 수정 중


### [Taeksan 소속](https://taeksan.co.kr/)
