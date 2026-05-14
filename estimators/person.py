"""
사람 기반 수위 추정기 (v5.0)

GPT 지적 [2] 수정:
  - pose keypoint threshold: 0.3 → 0.15 (하체 가려져도 검출)
  - 상대 침수율(%) 병행 출력 (GPT [1][6])
  - scale 이상값 clamp: 실제 사람 키 범위(100~200cm)만 허용
  - 자세 판별 신뢰도 반영
"""
import numpy as np
import torch
from typing import Optional, Tuple

from utils.config import BODY_HEIGHT_TABLE, KP
from utils.dataclasses import DepthEstimate, PersonMeta
from utils.image_utils import to_tensor, softmax_np, avg_valid
from confidence.calibration import (
    size_confidence, edge_confidence, person_pose_confidence,
    person_posture_quality, compute_final_confidence,
)

# 사람 키 scale 물리적 범위 (cm/px)
# 이미지에서 사람이 최소 20px~최대 이미지 전체 높이라 가정
SCALE_MIN_CM_PER_PX = 0.05   # 매우 큰 사람 (이미지 꽉 참)
SCALE_MAX_CM_PER_PX = 5.0    # 매우 작은 사람 (멀리 있음)
# 침수 깊이 물리적 상한
PERSON_DEPTH_MAX_CM = 200.0


def classify_person(crop, child_model, gender_model) -> PersonMeta:
    meta = PersonMeta()
    if crop.size == 0:
        return meta
    if child_model is not None:
        try:
            with torch.no_grad():
                p = softmax_np(child_model(to_tensor(crop)))
            meta.is_child   = bool(p[1] > p[0])
            meta.child_conf = float(max(p))
        except Exception:
            pass
    if not meta.is_child and gender_model is not None:
        try:
            with torch.no_grad():
                p = softmax_np(gender_model(to_tensor(crop)))
            meta.gender      = "female" if p[1] > p[0] else "male"
            meta.gender_conf = float(max(p))
        except Exception:
            pass
    if meta.is_child:
        meta.body_key = "child"
    elif meta.gender == "female":
        meta.body_key = "adult_female"
    else:
        meta.body_key = "adult_male"
    return meta


def estimate_depth_person(
    crop, bbox, kps_17x3, img_h, img_w,
    child_model, gender_model, ground_plane=None,
) -> Optional[DepthEstimate]:

    x1, y1, x2, y2 = [int(v) for v in bbox]
    bbox_h = y2 - y1
    if bbox_h < 20 or crop.size == 0:
        return None

    meta = classify_person(crop, child_model, gender_model)
    H    = BODY_HEIGHT_TABLE[meta.body_key]

    scale_cm_per_px = None
    depth_cm        = None
    uncertainty_cm  = 0.0
    method_parts    = []

    if kps_17x3 is not None:
        # GPT [2]: threshold 0.3 → 0.15 로 낮춤 (하체 가림 대응)
        def ky(idx): return float(kps_17x3[idx, 1]) if kps_17x3[idx, 2] > 0.15 else None
        def kconf(idx): return float(kps_17x3[idx, 2])

        nose_y     = ky(KP["nose"])
        shoulder_y = avg_valid(ky(KP["l_shoulder"]), ky(KP["r_shoulder"]))
        hip_y      = avg_valid(ky(KP["l_hip"]),      ky(KP["r_hip"]))
        knee_y     = avg_valid(ky(KP["l_knee"]),     ky(KP["r_knee"]))
        ankle_y    = avg_valid(ky(KP["l_ankle"]),    ky(KP["r_ankle"]))
        foot_y     = float(y2)

        # Scale calibration (두정점 > 어깨 > bbox 순 우선순위)
        if nose_y is not None and (foot_y - nose_y) > 20:
            raw_scale = H["total"] / (foot_y - nose_y)
            # 물리적 범위 clamp (GPT [1]: 비현실 scale 차단)
            scale_cm_per_px = float(np.clip(raw_scale, SCALE_MIN_CM_PER_PX, SCALE_MAX_CM_PER_PX))
            method_parts.append(f"scale=두정점({scale_cm_per_px:.3f}cm/px)")
        elif shoulder_y is not None and (foot_y - shoulder_y) > 10:
            raw_scale = H["shoulder"] / (foot_y - shoulder_y)
            scale_cm_per_px = float(np.clip(raw_scale, SCALE_MIN_CM_PER_PX, SCALE_MAX_CM_PER_PX))
            method_parts.append(f"scale=어깨({scale_cm_per_px:.3f}cm/px)")
        elif ground_plane and ground_plane.valid:
            scale_cm_per_px = ground_plane.scale_cm_per_px
            method_parts.append("scale=Homography")

        if scale_cm_per_px is None:
            scale_cm_per_px = float(np.clip(
                H["total"] / max(bbox_h, 1),
                SCALE_MIN_CM_PER_PX, SCALE_MAX_CM_PER_PX))
            method_parts.append("scale=bbox(fallback)")

        # 수위 계산
        candidates = []
        if ankle_y is not None:
            d = float(np.clip((foot_y - ankle_y) * scale_cm_per_px, 0, H["ankle"] * 2.0))
            candidates.append(("ankle", d, 1.5))
        if knee_y is not None and ankle_y is None:
            d = float(np.clip((foot_y - knee_y) * scale_cm_per_px, 0, H["knee"] * 1.3))
            candidates.append(("knee", d, 1.1))
        if hip_y is not None and ankle_y is None and knee_y is None:
            d = float(np.clip((foot_y - hip_y) * scale_cm_per_px, 0, H["hip"] * 1.1))
            candidates.append(("hip", d, 0.7))

        if candidates:
            total_w  = sum(w for _, _, w in candidates)
            depth_cm = sum(d * w for _, d, w in candidates) / total_w
            dvals    = [d for _, d, _ in candidates]
            uncertainty_cm = float(np.std(dvals)) if len(dvals) > 1 else depth_cm * 0.15
            method_parts.append("kp:" + "+".join(n for n, _, _ in candidates))

    posture, posture_quality = person_posture_quality(kps_17x3, bbox_h)

    if depth_cm is None:
        scale_cm_per_px = scale_cm_per_px or float(np.clip(
            H["total"] / max(bbox_h, 1), SCALE_MIN_CM_PER_PX, SCALE_MAX_CM_PER_PX))
        depth_cm       = 0.0
        uncertainty_cm = H["ankle"]
        method_parts.append("no_pose_fallback")

    # 최종 clamp
    depth_cm = float(np.clip(depth_cm, 0.0, PERSON_DEPTH_MAX_CM))

    # 상대 침수율 계산 (GPT [6])
    relative_pct = (depth_cm / H["shoulder"]) * 100.0 if H["shoulder"] > 0 else 0.0

    # Dynamic confidence
    sc    = size_confidence(bbox, img_h, img_w)
    ec    = edge_confidence(bbox, img_h, img_w)
    pc    = person_pose_confidence(kps_17x3,
                [KP["l_ankle"], KP["r_ankle"], KP["l_knee"], KP["r_knee"]])
    base  = meta.child_conf * 0.5 + meta.gender_conf * 0.3 + 0.2
    final = compute_final_confidence(base, sc, ec, pc * posture_quality)

    who   = f"{'아동' if meta.is_child else ('여성' if meta.gender=='female' else '남성')}"
    detail = (f"{who}({meta.body_key}) | {posture} | "
              f"침수율:{relative_pct:.0f}% | {' | '.join(method_parts)}")

    return DepthEstimate(
        source="person",
        depth_cm=depth_cm,
        confidence=final,
        detail=detail,
        bbox=(x1, y1, x2, y2),
        scale_cm_per_px=scale_cm_per_px,
        method=("scale_calibration"
                if any(x in str(method_parts) for x in ["두정점","어깨"])
                else "no_pose_fallback"),
        uncertainty_cm=uncertainty_cm,
    )