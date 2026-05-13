"""
전역 설정 및 상수  (v4.0)
"""

# ── 경로 ─────────────────────────────────────────────────────
MODEL_DIR  = "./model_folder"
OUTPUT_DIR = "./output"

# ── 모델 파일명 (변경된 파일명 반영) ─────────────────────────
MODEL_FILES = {
    "situation": "best_situation_classification_model_ver3_ResNet-18.pth",
    "child":     "child_classifier_EfficentNet-B0_ver2.pth",
    "gender":    "gender_classifier_EfficentNet-B0.pth",
    "car_cls":   "car_classification_EfficentNet-B0.pth",   # NEW
    "flood_seg": "flood_semtic_seg.keras",
    "yolo_main": "yolo11n.pt",
    "yolo_pose": "yolo11n-pose.pt",
    "yolo_car":  "car_part_segmentation.pt",
    "yolo_sign": "sign_detection_ver1.pt",
    "yolo_tube": "Tubular_Marker_detection.pt",
}

# ── 차량 분류 클래스 (EfficientNet-B0, 5cls) ─────────────────
CAR_CLS_NAMES = ["bus", "car-mini", "car-pickup", "car-sedan", "car-suv"]
# 내부 처리용 정규화 키
CAR_CLS_TO_KEY = {
    "bus":        "bus",
    "car-mini":   "mini",
    "car-pickup":  "pickup",
    "car-sedan":  "sedan",
    "car-suv":    "suv",
}

# ── 차량 종류별 지면 기준 부품 높이 (cm) ─────────────────────
#    출처: 제공된 차량 부위 높이 표 + 일반 차량 제원 기준
#    sedan/SUV는 제공 표 기준, 나머지는 일반 제원 평균
CAR_PART_HEIGHT_CM = {
    # 부품명: {차종: 높이(cm)}
    "Front bumper": {
        "sedan": 21.5, "suv": 30.0, "mini": 18.0,
        "pickup": 35.0, "bus": 40.0,
    },
    "Rear bumper": {
        "sedan": 21.5, "suv": 30.0, "mini": 18.0,
        "pickup": 35.0, "bus": 40.0,
    },
    "Headlight - -L-": {
        "sedan": 67.5, "suv": 90.0, "mini": 60.0,
        "pickup": 95.0, "bus": 115.0,
    },
    "Headlight - -R-": {
        "sedan": 67.5, "suv": 90.0, "mini": 60.0,
        "pickup": 95.0, "bus": 115.0,
    },
    "Car hood": {
        "sedan": 85.0, "suv": 107.5, "mini": 75.0,
        "pickup": 110.0, "bus": 130.0,
    },
    "Car boot": {
        "sedan": 77.5, "suv": 90.0, "mini": 70.0,
        "pickup": 85.0, "bus": 105.0,
    },
    "Side mirror - -L-": {
        "sedan": 105.0, "suv": 120.0, "mini": 95.0,
        "pickup": 130.0, "bus": 150.0,
    },
    "Side mirror - -R-": {
        "sedan": 105.0, "suv": 120.0, "mini": 95.0,
        "pickup": 130.0, "bus": 150.0,
    },
    "Fender - -F-L-": {
        "sedan": 75.0, "suv": 100.0, "mini": 68.0,
        "pickup": 105.0, "bus": 120.0,
    },
    "Fender - -F-R-": {
        "sedan": 75.0, "suv": 100.0, "mini": 68.0,
        "pickup": 105.0, "bus": 120.0,
    },
    "Fender - -R-L-": {
        "sedan": 75.0, "suv": 100.0, "mini": 68.0,
        "pickup": 105.0, "bus": 120.0,
    },
    "Fender - -R-R-": {
        "sedan": 75.0, "suv": 100.0, "mini": 68.0,
        "pickup": 105.0, "bus": 120.0,
    },
}

# ── 차량 전체 높이 (cm, 지면~루프) ───────────────────────────
#    제공 표: sedan 루프 140~155cm, SUV 165~185cm
CAR_TOTAL_HEIGHT_CM = {
    "sedan":  147.0,   # 제공 표 평균
    "suv":    175.0,   # 제공 표 평균
    "mini":   148.0,   # 경차 평균 (모닝 등)
    "pickup": 185.0,   # 픽업트럭 평균
    "bus":    310.0,   # 버스 평균
}

# ── YOLO cls id → 기본 차종 매핑 (EfficientNet 없을 때 fallback) ─
YOLO_CLS_TO_CAR = {
    2: "sedan",   # car
    5: "bus",     # bus
    7: "pickup",  # truck
}

# ── COCO 17-keypoint 인덱스 ───────────────────────────────────
KP = {
    "nose": 0,
    "l_eye": 1, "r_eye": 2,
    "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow": 7,     "r_elbow": 8,
    "l_wrist": 9,     "r_wrist": 10,
    "l_hip": 11,      "r_hip": 12,
    "l_knee": 13,     "r_knee": 14,
    "l_ankle": 15,    "r_ankle": 16,
}

# ── 신체 치수 테이블 (cm, 바닥 기준) ─────────────────────────
#    출처: 제공된 한국인 평균 신체 치수 표
BODY_HEIGHT_TABLE = {
    "adult_male": {
        "total": 171.0, "head": 171.0,
        "shoulder": 141.5, "hip": 94.0,
        "knee": 49.0, "ankle": 8.0,
    },
    "adult_female": {
        "total": 158.0, "head": 158.0,
        "shoulder": 130.5, "hip": 85.5,
        "knee": 45.0, "ankle": 7.0,
    },
    "child": {
        "total": 120.0, "head": 120.0,
        "shoulder": 95.0, "hip": 60.0,
        "knee": 32.0, "ankle": 5.0,
    },
}

# ── 도로 시설물 기준 높이 ─────────────────────────────────────
SIGN_HEIGHT_CM    = 320.0   # 도로 표지판
TUBULAR_HEIGHT_CM =  80.0   # 시선유도봉

# ── 침수 위험 단계 ────────────────────────────────────────────
FLOOD_LEVELS = [
    (  0,  10, "매우 낮음 💧", "#64B5F6"),
    ( 10,  30, "낮음 🌊",     "#4CAF50"),
    ( 30,  50, "주의 ⚠️",     "#FDD835"),
    ( 50,  80, "경고 🔶",     "#FF9800"),
    ( 80, 120, "위험 🚨",     "#F44336"),
    (120, 999, "매우 위험 ☠️","#9C27B0"),
]

# ── Fusion 기본 소스 가중치 ──────────────────────────────────
BASE_SOURCE_WEIGHTS = {
    "person":    1.5,
    "car":       1.4,   # EfficientNet 차종 분류 추가로 신뢰도 상승
    "tubular":   1.2,
    "sign":      0.8,
    "flood_seg": 0.6,
}

# ── 추정 방법별 품질 가중치 ──────────────────────────────────
METHOD_QUALITY = {
    "scale_calibration":  1.3,
    "car_part_seg":       1.3,   # 차종분류 정확해졌으므로 상향
    "ransac_waterline":   1.1,
    "sign_ratio":         0.9,
    "tubular_ratio":      1.0,
    "car_bbox_fallback":  0.5,
    "no_pose_fallback":   0.4,
    "unknown":            0.8,
}

# ── 시각화 ───────────────────────────────────────────────────
OVERLAY_COLORS = {
    "person":    ( 50, 220, 120),
    "car":       (255, 200,  30),
    "sign":      ( 30, 180, 255),
    "tubular":   (220, 100, 220),
    "flood_seg": (100, 200, 255),
}
OVERLAY_ICONS = {
    "person":    "👤",
    "car":       "🚗",
    "sign":      "🚧",
    "tubular":   "🏮",
    "flood_seg": "🗺️",
}
