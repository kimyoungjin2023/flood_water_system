"""
도로 시설물 기반 수위 추정기 (v4.0)
- 표지판 (3.2m 기준)
- 시선유도봉 (80cm 기준)
"""
import numpy as np
from typing import Tuple

from utils.config import SIGN_HEIGHT_CM, TUBULAR_HEIGHT_CM
from utils.dataclasses import DepthEstimate
from confidence.calibration import (
    size_confidence, edge_confidence, compute_final_confidence,
)


def _submerged_ratio(y2: float, img_h: int,
                     obj_h_px: float) -> Tuple[float, float]:
    """
    객체 하단 y2 기반 침수 비율 추정
    y2가 img_h에 가까울수록 침수 없음
    """
    bottom_ratio = y2 / max(img_h, 1)
    if bottom_ratio >= 0.93:
        return 0.0, 0.35

    hidden_px   = img_h - y2
    full_h_px   = obj_h_px + hidden_px
    sub_r       = max(0.0, min(hidden_px / max(full_h_px, 1), 0.95))
    conf        = 0.75 if 0.05 < sub_r < 0.85 else 0.45
    return sub_r, conf


def estimate_depth_sign(bbox: Tuple, img_h: int, img_w: int) -> DepthEstimate:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    sign_h_px = max(y2 - y1, 1)

    sub_r, base_conf = _submerged_ratio(float(y2), img_h, sign_h_px)
    depth_cm = float(np.clip(SIGN_HEIGHT_CM * sub_r, 0, SIGN_HEIGHT_CM))

    sc = size_confidence(bbox, img_h, img_w)
    ec = edge_confidence(bbox, img_h, img_w)
    conf = compute_final_confidence(base_conf, sc, ec)

    return DepthEstimate(
        source="sign",
        depth_cm=depth_cm,
        confidence=conf,
        detail=(f"표지판 {SIGN_HEIGHT_CM:.0f}cm | "
                f"하단:{float(y2)/img_h:.2%} | 침수:{sub_r:.1%} → {depth_cm:.1f}cm"),
        bbox=(x1, y1, x2, y2),
        method="sign_ratio",
        uncertainty_cm=depth_cm * 0.15,
    )


def estimate_depth_tubular(bbox: Tuple, img_h: int, img_w: int) -> DepthEstimate:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    tube_h_px = max(y2 - y1, 1)

    sub_r, base_conf = _submerged_ratio(float(y2), img_h, tube_h_px)
    depth_cm = float(np.clip(TUBULAR_HEIGHT_CM * sub_r, 0, TUBULAR_HEIGHT_CM))

    sc = size_confidence(bbox, img_h, img_w)
    ec = edge_confidence(bbox, img_h, img_w)
    conf = compute_final_confidence(base_conf, sc, ec)

    return DepthEstimate(
        source="tubular",
        depth_cm=depth_cm,
        confidence=conf,
        detail=(f"시선유도봉 {TUBULAR_HEIGHT_CM:.0f}cm | "
                f"하단:{float(y2)/img_h:.2%} | 침수:{sub_r:.1%} → {depth_cm:.1f}cm"),
        bbox=(x1, y1, x2, y2),
        method="tubular_ratio",
        uncertainty_cm=depth_cm * 0.12,
    )
