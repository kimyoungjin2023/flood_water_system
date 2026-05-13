"""
사람 기반 수위 추정기 (v4.0)

개선사항:
  [2] Pose visibility confidence → 동적 신뢰도
  [2] 자세(standing/sitting/crouching) 판별 → 신뢰도 보정
  [5] Dynamic Confidence (bbox 크기, 경계 근접도)
  [1] GroundPlane scale 사용 + 객체로 scale 보정
"""
import numpy as np
import torch
from typing import Optional, Tuple

from utils.config import BODY_HEIGHT_TABLE, KP
from utils.dataclasses import DepthEstimate, PersonMeta, GroundPlane
from utils.image_utils import to_tensor, softmax_np, avg_valid
from confidence.calibration import (
    size_confidence, edge_confidence, person_pose_confidence,
    person_posture_quality, compute_final_confidence,
)
from geometry.ground_plane import refine_with_object


def classify_person(crop: np.ndarray,
                    child_model, gender_model) -> PersonMeta:
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
    crop:           np.ndarray,
    bbox:           Tuple,
    kps_17x3:       Optional[np.ndarray],
    img_h:          int,
    img_w:          int,
    child_model,
    gender_model,
    ground_plane:   Optional[GroundPlane] = None,
) -> Optional[DepthEstimate]:
    """
    사람 기반 수위 추정

    수위 계산 원리:
      발끝(bbox 하단) ~ 가장 낮은 visible keypoint 픽셀 거리
      × 키 기반 cm/px scale = 침수 깊이

    신뢰도 결정 인자 (Dynamic Confidence):
      1) 사람 크기 (bbox 면적 비율)
      2) 이미지 경계 근접도
      3) 하체 keypoint visibility score
      4) 자세 (서있음/앉음/웅크림)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bbox_h = y2 - y1
    if bbox_h < 30 or crop.size == 0:
        return None

    # ── 속성 분류 ─────────────────────────────────────────────
    meta = classify_person(crop, child_model, gender_model)
    H    = BODY_HEIGHT_TABLE[meta.body_key]

    # ── Pose 분석 ─────────────────────────────────────────────
    scale_cm_per_px = None
    depth_cm        = None
    uncertainty_cm  = 0.0
    method_parts    = []

    if kps_17x3 is not None:
        def ky(idx): return float(kps_17x3[idx, 1]) if kps_17x3[idx, 2] > 0.3 else None

        nose_y     = ky(KP["nose"])
        shoulder_y = avg_valid(ky(KP["l_shoulder"]), ky(KP["r_shoulder"]))
        hip_y      = avg_valid(ky(KP["l_hip"]),      ky(KP["r_hip"]))
        knee_y     = avg_valid(ky(KP["l_knee"]),     ky(KP["r_knee"]))
        ankle_y    = avg_valid(ky(KP["l_ankle"]),    ky(KP["r_ankle"]))
        foot_y     = float(y2)

        # ── Scale calibration (ChatGPT [2] 핵심) ──────────────
        if nose_y is not None and (foot_y - nose_y) > 20:
            pixel_h         = foot_y - nose_y
            scale_cm_per_px = H["total"] / pixel_h
            method_parts.append(f"scale=두정점({scale_cm_per_px:.3f}cm/px)")
        elif shoulder_y is not None and (foot_y - shoulder_y) > 10:
            pixel_h         = foot_y - shoulder_y
            scale_cm_per_px = H["shoulder"] / pixel_h
            method_parts.append(f"scale=어깨({scale_cm_per_px:.3f}cm/px)")
        elif ground_plane and ground_plane.valid:
            scale_cm_per_px = ground_plane.scale_cm_per_px
            method_parts.append("scale=Homography")
        else:
            scale_cm_per_px = H["total"] / max(bbox_h, 1)
            method_parts.append("scale=bbox(fallback)")

        # GroundPlane scale 보정 (사람으로 갱신)
        if (ground_plane and ground_plane.valid
                and scale_cm_per_px
                and not ground_plane.calibrated):
            ground_plane = refine_with_object(
                ground_plane,
                float(bbox_h),
                scale_cm_per_px * bbox_h,
                source="person",
            )

        # ── 수위 계산 ─────────────────────────────────────────
        candidates = []
        if ankle_y is not None:
            d = max(0.0, (foot_y - ankle_y) * scale_cm_per_px)
            d = min(d, H["ankle"] * 2.0)
            candidates.append(("ankle", d, 1.5))
        if knee_y is not None and ankle_y is None:
            d = max(0.0, (foot_y - knee_y) * scale_cm_per_px)
            d = min(d, H["knee"] * 1.3)
            candidates.append(("knee", d, 1.1))
        if hip_y is not None and ankle_y is None and knee_y is None:
            d = max(0.0, (foot_y - hip_y) * scale_cm_per_px)
            d = min(d, H["hip"] * 1.1)
            candidates.append(("hip", d, 0.7))

        if candidates:
            total_w  = sum(w for _, _, w in candidates)
            depth_cm = sum(d * w for _, d, w in candidates) / total_w
            depth_vals = [d for _, d, _ in candidates]
            uncertainty_cm = float(np.std(depth_vals)) if len(depth_vals) > 1 else depth_cm * 0.15
            method_parts.append("kp:" + "+".join(n for n, _, _ in candidates))

    # ── 자세 판별 (ChatGPT [2]) ────────────────────────────────
    posture, posture_quality = person_posture_quality(kps_17x3, bbox_h)

    # ── Pose 없을 때 fallback ─────────────────────────────────
    if depth_cm is None:
        scale_cm_per_px = scale_cm_per_px or (H["total"] / max(bbox_h, 1))
        depth_cm        = max(0.0, (img_h - y2) / max(img_h, 1) * H["ankle"])
        uncertainty_cm  = depth_cm * 0.35
        method_parts.append("no_pose_fallback")

    depth_cm = float(np.clip(depth_cm, 0.0, H["shoulder"]))

    # ── Dynamic Confidence (ChatGPT [5]) ─────────────────────
    sc    = size_confidence(bbox, img_h, img_w)
    ec    = edge_confidence(bbox, img_h, img_w)
    pc    = person_pose_confidence(
                kps_17x3,
                [KP["l_ankle"], KP["r_ankle"], KP["l_knee"], KP["r_knee"]])
    base  = meta.child_conf * 0.5 + meta.gender_conf * 0.3 + 0.2
    final = compute_final_confidence(base, sc, ec, pc * posture_quality)

    who = f"{'아동' if meta.is_child else ('여성' if meta.gender=='female' else '남성')}"
    detail = (f"{who}({meta.body_key}) | {posture} | "
              f"{' | '.join(method_parts)}")

    return DepthEstimate(
        source="person",
        depth_cm=depth_cm,
        confidence=final,
        detail=detail,
        bbox=(x1, y1, x2, y2),
        scale_cm_per_px=scale_cm_per_px,
        method="scale_calibration" if "두정점" in str(method_parts) or "어깨" in str(method_parts)
               else "no_pose_fallback",
        uncertainty_cm=uncertainty_cm,
    )
