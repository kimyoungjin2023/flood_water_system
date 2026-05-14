"""
가중 융합 모듈 (v5.0)

GPT 지적 [1][8] 수정:
  - hard clipping: 물리적으로 불가능한 수위(>300cm) 제거
  - max_influence: 한 소스가 결과를 독점하지 못하도록 기여도 상한 제한
  - robust median fallback: extreme outlier 발생 시 중앙값 사용
  - IQR k 값 강화 (1.5 → 1.2): 더 엄격한 outlier 제거
"""
import numpy as np
from typing import List, Tuple

from utils.config import BASE_SOURCE_WEIGHTS, METHOD_QUALITY, FLOOD_LEVELS
from utils.dataclasses import DepthEstimate

DEPTH_MIN_CM  =   0.0
DEPTH_MAX_CM  = 250.0   # 883cm 같은 폭주 차단
MAX_INFLUENCE =   0.60  # 단일 소스 최대 기여 비율


def _source_weight(est: DepthEstimate) -> float:
    base    = BASE_SOURCE_WEIGHTS.get(est.source, 1.0)
    quality = METHOD_QUALITY.get(est.method, 0.8)
    if est.depth_cm > 0 and est.uncertainty_cm > 0:
        unc_factor = max(1.0 - (est.uncertainty_cm / est.depth_cm) * 0.5, 0.3)
    else:
        unc_factor = 1.0
    return base * quality * est.confidence * unc_factor


def hard_clip(estimates: List[DepthEstimate]) -> Tuple[List, List]:
    valid   = [e for e in estimates if DEPTH_MIN_CM <= e.depth_cm <= DEPTH_MAX_CM]
    clipped = [e for e in estimates if e.depth_cm < DEPTH_MIN_CM or e.depth_cm > DEPTH_MAX_CM]
    if clipped:
        print(f"  [Fusion] Hard clip 제거 ({len(clipped)}개): "
              f"{[f'{e.source}:{e.depth_cm:.1f}cm' for e in clipped]}")
    return (valid, clipped) if valid else (estimates, [])


def remove_outliers_iqr(estimates: List[DepthEstimate], k: float = 1.2) -> Tuple[List, List]:
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
    if not estimates:
        return 0.0, 0.0, 0.0, {}, [], []

    valid, hard_removed = hard_clip(estimates)
    inliers, iqr_removed = remove_outliers_iqr(valid)
    all_outliers = hard_removed + iqr_removed

    if iqr_removed:
        print(f"  [Fusion] IQR 제거 ({len(iqr_removed)}개): "
              f"{[f'{e.source}:{e.depth_cm:.1f}cm' for e in iqr_removed]}")

    raw_weights = {i: _source_weight(e) for i, e in enumerate(inliers)}
    total_w     = sum(raw_weights.values())

    if total_w < 1e-9:
        avg = float(np.median([e.depth_cm for e in inliers]))
        return avg, avg, avg * 0.3, {}, inliers, all_outliers

    # max_influence 제한
    capped = {}
    for i, w in raw_weights.items():
        capped[i] = min(w, MAX_INFLUENCE * total_w)
    total_capped = sum(capped.values())

    avg      = float(np.mean([e.depth_cm for e in inliers]))
    weighted = sum(inliers[i].depth_cm * w for i, w in capped.items()) / total_capped

    if len(inliers) == 1:
        weighted *= 0.75

    # robust median fallback
    depths_arr = np.array([e.depth_cm for e in inliers])
    median_val = float(np.median(depths_arr))
    if len(inliers) >= 3 and abs(weighted - median_val) > median_val * 0.8:
        print(f"  [Fusion] robust median fallback: {weighted:.1f}cm → {median_val:.1f}cm")
        weighted = median_val

    weighted    = float(np.clip(weighted, DEPTH_MIN_CM, DEPTH_MAX_CM))
    avg         = float(np.clip(avg,      DEPTH_MIN_CM, DEPTH_MAX_CM))
    depth_std   = float(np.std(depths_arr)) if len(inliers) > 1 else 0.0
    unc_mean    = float(np.mean([e.uncertainty_cm for e in inliers]))
    uncertainty = float(np.clip((depth_std + unc_mean) / 2.0, 0, DEPTH_MAX_CM * 0.5))

    weight_dict = {e.source: round(capped[i] / total_capped, 3)
                   for i, e in enumerate(inliers)}
    return avg, weighted, uncertainty, weight_dict, inliers, all_outliers


def get_flood_level(depth_cm: float) -> Tuple[str, str]:
    for lo, hi, label, color in FLOOD_LEVELS:
        if lo <= depth_cm < hi:
            return label, color
    return FLOOD_LEVELS[-1][2], FLOOD_LEVELS[-1][3]