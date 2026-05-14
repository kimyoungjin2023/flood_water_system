"""
차량 기반 수위 추정기 (v5.0)

GPT 지적 [3] 수정:
  - 단일 평균값 대신 (min, max) 범위 사용 → uncertainty 반영
  - px_per_cm 계산에 차량 bbox가 화면에 꽉 찬 경우 보정
  - 상대 침수율 (차량 높이 대비 %) 병행 출력
  - 차량 깊이 물리적 상한 clamp
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

# 차량 높이 범위 (min, max) — 단일값 대신 범위로 uncertainty 계산 (GPT [3])
CAR_TOTAL_HEIGHT_RANGE = {
    "sedan":  (135.0, 160.0),
    "suv":    (160.0, 195.0),
    "mini":   (140.0, 165.0),
    "pickup": (175.0, 210.0),
    "bus":    (280.0, 340.0),
}
CAR_DEPTH_MAX_CM = 350.0  # 버스 높이 이상은 비현실


def classify_car(crop, yolo_cls_id, car_cls_model, bbox, img_h, img_w) -> CarMeta:
    meta = CarMeta(yolo_cls_id=yolo_cls_id)
    if car_cls_model is not None and crop.size > 0:
        try:
            with torch.no_grad():
                p = softmax_np(car_cls_model(to_tensor(crop)))
            cls_idx            = int(np.argmax(p))
            meta.car_type      = CAR_CLS_TO_KEY[CAR_CLS_NAMES[cls_idx]]
            meta.car_type_conf = float(p[cls_idx])
        except Exception:
            meta.car_type      = YOLO_CLS_TO_CAR.get(yolo_cls_id, "sedan")
            meta.car_type_conf = 0.40
    else:
        meta.car_type      = YOLO_CLS_TO_CAR.get(yolo_cls_id, "sedan")
        meta.car_type_conf = 0.40
    meta.orientation, _ = car_orientation_from_bbox(bbox, img_h, img_w)
    return meta


def estimate_depth_car(
    car_bbox, yolo_cls_id, seg_result, img_h, img_w,
    car_cls_model=None, crop=None, ground_plane=None,
) -> DepthEstimate:

    x1, y1, x2, y2 = [int(v) for v in car_bbox]
    car_h_px = float(y2 - y1)
    if car_h_px < 10:
        return DepthEstimate("car", 0.0, 0.10, "bbox 너무 작음", car_bbox)

    import numpy as np
    _crop     = crop if (crop is not None and crop.size > 0) else np.zeros((4,4,3), np.uint8)
    meta      = classify_car(_crop, yolo_cls_id, car_cls_model, car_bbox, img_h, img_w)
    car_type  = meta.car_type

    # GPT [3]: 범위(min/max) 사용
    h_min, h_max  = CAR_TOTAL_HEIGHT_RANGE[car_type]
    car_real_h    = (h_min + h_max) / 2.0
    h_uncertainty = (h_max - h_min) / 2.0  # 차종 높이 자체의 불확실도
    px_per_cm     = car_h_px / car_real_h

    side_only  = {"Side mirror - -L-", "Side mirror - -R-",
                  "Fender - -F-L-", "Fender - -F-R-",
                  "Fender - -R-L-", "Fender - -R-R-"}
    front_rear = {"Front bumper","Rear bumper",
                  "Headlight - -L-","Headlight - -R-","Car hood"}

    depth_list, used_parts = [], []

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
            orient_penalty = 1.0
            if meta.orientation == "front_rear" and part_name in side_only:
                orient_penalty = 0.4
            elif meta.orientation == "side" and part_name in front_rear:
                orient_penalty = 0.5

            _, _, _, by2 = box.xyxy[0].tolist()
            h_above_cm   = (car_h_px - float(by2)) / max(px_per_cm, 0.01)

            matched_h = None
            for p_key, p_h in CAR_PART_HEIGHT_CM.items():
                if (p_key.split(" -")[0].lower() in part_name.lower()
                        or part_name.lower() in p_key.lower()):
                    matched_h = p_h.get(car_type, p_h.get("sedan", 20.0))
                    break
            if matched_h is None:
                continue

            flood_d = float(np.clip(matched_h - h_above_cm, 0, CAR_DEPTH_MAX_CM))
            depth_list.append((flood_d, conf_part * orient_penalty, part_name))
            used_parts.append(part_name)

    if depth_list:
        total_w  = sum(c for _, c, _ in depth_list)
        depth_cm = sum(d * c for d, c, _ in depth_list) / max(total_w, 1e-9)
        # 부품별 편차 + 차종 높이 불확실도 합산
        part_std  = float(np.std([d for d, _, _ in depth_list])) if len(depth_list) > 1 else 5.0
        uncertainty_cm = part_std + h_uncertainty * 0.3
        base_conf = min(total_w / len(depth_list), 1.0)
        method    = "car_part_seg"
        parts_str = ", ".join(set(used_parts[:4]))
    else:
        depth_cm       = car_real_h * 0.10
        uncertainty_cm = car_real_h * 0.10 + h_uncertainty
        parts_str      = "없음"
        base_conf      = 0.15
        method         = "car_bbox_fallback"

    depth_cm = float(np.clip(depth_cm, 0, CAR_DEPTH_MAX_CM))

    # 상대 침수율 (차량 전체 높이 대비)
    relative_pct = (depth_cm / car_real_h) * 100.0

    sc    = size_confidence(car_bbox, img_h, img_w)
    ec    = edge_confidence(car_bbox, img_h, img_w)
    vc    = car_visibility_confidence(seg_result, car_bbox, img_h)
    final = compute_final_confidence(base_conf, sc, ec, vc * meta.car_type_conf)

    detail = (f"{car_type}({meta.car_type_conf:.0%}) | "
              f"orient:{meta.orientation} | "
              f"침수율:{relative_pct:.0f}% | 부품:{parts_str}")

    return DepthEstimate(
        source="car",
        depth_cm=depth_cm,
        confidence=final,
        detail=detail,
        bbox=(x1, y1, x2, y2),
        method=method,
        uncertainty_cm=float(np.clip(uncertainty_cm, 0, CAR_DEPTH_MAX_CM * 0.5)),
    )