"""
침수 Semantic Segmentation 기반 수위 추정기 (v4.0)
"""
import numpy as np
from typing import Optional

from utils.dataclasses import DepthEstimate, WaterlineResult
from utils.config import BODY_HEIGHT_TABLE


def estimate_depth_seg(
    waterline:   WaterlineResult,
    img_h:       int,
    img_w:       int,
    scale_hint:  Optional[float] = None,
) -> Optional[DepthEstimate]:
    """
    RANSAC 수면선 → 수위 추정

    scale_hint: 다른 소스(사람/차량)에서 calibration된 cm/px scale
    없으면 성인 평균 키 기준 상대값 (낮은 신뢰도)
    """
    if not waterline.valid:
        return None

    if scale_hint and scale_hint > 0:
        pixels_flooded = img_h - waterline.waterline_y
        depth_cm = pixels_flooded * scale_hint
        method   = f"RANSAC+scale({scale_hint:.3f}cm/px)"
        uncertainty = depth_cm * 0.20
    else:
        ref_h    = BODY_HEIGHT_TABLE["adult_male"]["shoulder"]
        depth_cm = waterline.flood_ratio * ref_h
        method   = "RANSAC+상대비율(참고)"
        uncertainty = depth_cm * 0.35   # scale 없으면 불확실도 높음

    depth_cm = max(0.0, depth_cm)

    # 신뢰도: RANSAC inlier × 픽셀 커버리지 (최대 0.65 — 절대 기준 없음)
    conf = min(
        waterline.inlier_ratio * 0.6 + waterline.pixel_ratio * 0.4,
        0.65,
    )

    return DepthEstimate(
        source="flood_seg",
        depth_cm=depth_cm,
        confidence=conf,
        detail=(f"RANSAC waterline y={waterline.waterline_y:.0f}px | "
                f"inlier:{waterline.inlier_ratio:.0%} | "
                f"flood:{waterline.flood_ratio:.1%} | {method}"),
        bbox=None,
        scale_cm_per_px=scale_hint,
        method="ransac_waterline",
        uncertainty_cm=float(uncertainty),
    )
