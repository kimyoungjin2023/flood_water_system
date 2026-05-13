"""
차량 기반 수위 추정기 (v4.0)

개선사항:
  [3] EfficientNet-B0 차량 종류 분류 모델 사용
       bus / car-mini / car-pickup / car-sedan / car-suv (5cls)
  [3] Vehicle orientation 추정 (front/rear/side)
  [5] Dynamic Confidence (부품 감지 수, bbox 크기)
"""
import numpy as np
import torch
from typing import Optional, Tuple

from utils.config import (
    CAR_CLS_NAMES, CAR_CLS_TO_KEY, CAR_PART_HEIGHT_CM,
    CAR_TOTAL_HEIGHT_CM, YOLO_CLS_TO_CAR,
)
from utils.dataclasses import DepthEstimate, CarMeta
from utils.image_utils import to_tensor, softmax_np
from confidence.calibration import (
    size_confidence, edge_confidence, car_orientation_from_bbox,
    car_visibility_confidence, compute_final_confidence,
)


def classify_car(crop: np.ndarray,
                 yolo_cls_id: int,
                 car_cls_model,
                 bbox: Tuple,
                 img_h: int, img_w: int) -> CarMeta:
    """
    차량 종류 분류

    우선순위:
    1) EfficientNet-B0 모델 (bus/mini/pickup/sedan/suv)  ← 가장 정확
    2) YOLO cls_id fallback (car=sedan/bus=bus/truck=pickup)
    3) bbox aspect ratio fallback (sedan vs suv 구분)
    """
    meta = CarMeta(yolo_cls_id=yolo_cls_id)

    # ── EfficientNet 분류 ─────────────────────────────────────
    if car_cls_model is not None and crop.size > 0:
        try:
            with torch.no_grad():
                p = softmax_np(car_cls_model(to_tensor(crop)))
            cls_idx           = int(np.argmax(p))
            meta.car_type     = CAR_CLS_TO_KEY[CAR_CLS_NAMES[cls_idx]]
            meta.car_type_conf = float(p[cls_idx])
        except Exception:
            meta.car_type      = YOLO_CLS_TO_CAR.get(yolo_cls_id, "sedan")
            meta.car_type_conf = 0.45
    else:
        # YOLO cls fallback
        meta.car_type      = YOLO_CLS_TO_CAR.get(yolo_cls_id, "sedan")
        meta.car_type_conf = 0.40

    # ── Orientation 추정 (ChatGPT [3]) ───────────────────────
    meta.orientation, _ = car_orientation_from_bbox(bbox, img_h, img_w)

    return meta


def estimate_depth_car(
    car_bbox:     Tuple,
    yolo_cls_id:  int,
    seg_result,
    img_h:        int,
    img_w:        int,
    car_cls_model = None,
    crop:         Optional[np.ndarray] = None,
    ground_plane  = None,
) -> DepthEstimate:
    """
    차량 기반 수위 추정 (v4.0)

    수위 계산 원리:
      1. EfficientNet으로 차종 분류 → 부품별 실제 높이(cm) 선택
      2. 차량 부품 Seg bbox 하단 y → 지면 기준 높이 계산
      3. 부품 실제 높이 - 현재 보이는 높이 = 침수 깊이

    orientation 보정:
      - side view: front bumper / headlight 등 신뢰도 높음
      - front/rear view: side mirror 등은 부정확 → 해당 부품 제외
    """
    x1, y1, x2, y2 = [int(v) for v in car_bbox]
    car_h_px = float(y2 - y1)
    if car_h_px < 10:
        return DepthEstimate("car", 0.0, 0.10, "bbox 너무 작음", car_bbox)

    # 차종 분류
    _crop     = crop if (crop is not None and crop.size > 0) else np.zeros((4,4,3), np.uint8)
    meta      = classify_car(_crop, yolo_cls_id, car_cls_model, car_bbox, img_h, img_w)
    car_type  = meta.car_type
    car_real_h = CAR_TOTAL_HEIGHT_CM[car_type]
    px_per_cm  = car_h_px / car_real_h

    depth_list = []
    used_parts = []

    # orientation에 따라 신뢰도가 낮은 부품 필터링
    side_only_parts = {"Side mirror - -L-", "Side mirror - -R-",
                       "Fender - -F-L-", "Fender - -F-R-",
                       "Fender - -R-L-", "Fender - -R-R-"}
    front_rear_parts = {"Front bumper", "Rear bumper",
                        "Headlight - -L-", "Headlight - -R-", "Car hood"}

    if (seg_result is not None
            and hasattr(seg_result, "boxes")
            and len(seg_result.boxes) > 0):

        names = seg_result.names
        for box in seg_result.boxes:
            cls_id    = int(box.cls[0])
            conf_part = float(box.conf[0])
            part_name = names[cls_id]
            if conf_part < 0.25:
                continue

            # Orientation 필터: front/rear 뷰에서 side-only 부품 신뢰도 하락
            orient_penalty = 1.0
            if meta.orientation == "front_rear" and part_name in side_only_parts:
                orient_penalty = 0.4
            elif meta.orientation == "side" and part_name in front_rear_parts:
                orient_penalty = 0.5

            _, _, _, by2 = box.xyxy[0].tolist()
            h_above_cm   = (car_h_px - float(by2)) / max(px_per_cm, 0.01)

            # 부품 실제 높이 매칭
            matched_h = None
            for p_key, p_h in CAR_PART_HEIGHT_CM.items():
                if (p_key.split(" -")[0].lower() in part_name.lower()
                        or part_name.lower() in p_key.lower()):
                    matched_h = p_h.get(car_type, p_h.get("sedan", 20.0))
                    break

            if matched_h is None:
                continue

            flood_d = max(0.0, matched_h - h_above_cm)
            adj_conf = conf_part * orient_penalty
            depth_list.append((flood_d, adj_conf, part_name))
            used_parts.append(part_name)

    # ── 집계 ─────────────────────────────────────────────────
    if depth_list:
        total_w  = sum(c for _, c, _ in depth_list)
        depth_cm = sum(d * c for d, c, _ in depth_list) / max(total_w, 1e-9)
        uncertainty_cm = float(np.std([d for d, _, _ in depth_list])) if len(depth_list) > 1 else 5.0
        parts_str = ", ".join(set(used_parts[:4]))
        base_conf = min(total_w / len(depth_list), 1.0)
        method    = "car_part_seg"
    else:
        depth_cm       = car_real_h * 0.10
        uncertainty_cm = car_real_h * 0.05
        parts_str      = "없음"
        base_conf      = 0.20
        method         = "car_bbox_fallback"

    # Dynamic Confidence
    sc    = size_confidence(car_bbox, img_h, img_w)
    ec    = edge_confidence(car_bbox, img_h, img_w)
    vc    = car_visibility_confidence(seg_result, car_bbox, img_h)
    final = compute_final_confidence(base_conf, sc, ec, vc * meta.car_type_conf)

    detail = (f"{car_type}({meta.car_type_conf:.0%}) | "
              f"orientation:{meta.orientation} | 부품:{parts_str}")

    return DepthEstimate(
        source="car",
        depth_cm=float(np.clip(depth_cm, 0, car_real_h)),
        confidence=final,
        detail=detail,
        bbox=(x1, y1, x2, y2),
        method=method,
        uncertainty_cm=uncertainty_cm,
    )
