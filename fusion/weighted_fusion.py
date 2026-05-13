"""
가중 융합 모듈 (v4.0)

개선사항:
  - IQR outlier 제거
  - Method quality 가중치
  - Uncertainty-aware fusion (불확실도 높은 추정치 자동 하향)
  - 단일 소스 신뢰도 할인
"""
import numpy as np
from typing import List, Tuple

from utils.config import BASE_SOURCE_WEIGHTS, METHOD_QUALITY, FLOOD_LEVELS
from utils.dataclasses import DepthEstimate


def _source_weight(est: DepthEstimate) -> float:
    base    = BASE_SOURCE_WEIGHTS.get(est.source, 1.0)
    quality = METHOD_QUALITY.get(est.method, 0.8)
    # 불확실도 보정: uncertainty가 depth의 50% 이상이면 가중치 감소
    if est.depth_cm > 0 and est.uncertainty_cm > 0:
        unc_ratio = est.uncertainty_cm / est.depth_cm
        unc_factor = max(1.0 - unc_ratio * 0.5, 0.3)
    else:
        unc_factor = 1.0
    return base * quality * est.confidence * unc_factor


def remove_outliers_iqr(estimates: List[DepthEstimate],
                         k: float = 1.5) -> Tuple[List, List]:
    if len(estimates) < 3:
        return estimates, []

    depths = np.array([e.depth_cm for e in estimates])
    q1, q3 = np.percentile(depths, [25, 75])
    iqr     = q3 - q1
    lo, hi  = q1 - k * iqr, q3 + k * iqr

    inliers  = [e for e in estimates if lo <= e.depth_cm <= hi]
    outliers = [e for e in estimates if e.depth_cm < lo or e.depth_cm > hi]
    return (inliers, outliers) if inliers else (estimates, [])


def fuse(estimates: List[DepthEstimate]) -> Tuple[float, float, float, dict, List, List]:
    """
    Returns:
      avg_depth, weighted_depth, uncertainty, weight_dict, inliers, outliers
    """
    if not estimates:
        return 0.0, 0.0, 0.0, {}, [], []

    inliers, outliers = remove_outliers_iqr(estimates)

    weights   = {i: _source_weight(e) for i, e in enumerate(inliers)}
    total_w   = sum(weights.values())

    if total_w < 1e-9:
        avg = float(np.mean([e.depth_cm for e in inliers]))
        return avg, avg, avg * 0.3, {}, inliers, outliers

    avg      = float(np.mean([e.depth_cm for e in inliers]))
    weighted = sum(inliers[i].depth_cm * w for i, w in weights.items()) / total_w

    # 단일 소스 할인
    if len(inliers) == 1:
        weighted *= 0.80

    # 불확실도: 추정치 분산 + 개별 uncertainty 가중 평균
    depth_std = float(np.std([e.depth_cm for e in inliers]))
    unc_mean  = float(np.mean([e.uncertainty_cm for e in inliers]))
    uncertainty = (depth_std + unc_mean) / 2.0

    weight_dict = {e.source: round(weights[i], 3) for i, e in enumerate(inliers)}
    return avg, weighted, uncertainty, weight_dict, inliers, outliers


def get_flood_level(depth_cm: float) -> Tuple[str, str]:
    for lo, hi, label, color in FLOOD_LEVELS:
        if lo <= depth_cm < hi:
            return label, color
    return FLOOD_LEVELS[-1][2], FLOOD_LEVELS[-1][3]
