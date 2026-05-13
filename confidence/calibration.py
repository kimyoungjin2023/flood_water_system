"""
신뢰도 보정 모듈 (v4.0)
(ChatGPT 피드백 [2][5][6] 반영 — confidence 독립 모듈화)

Dynamic Confidence Calibration:
  bbox 크기, 이미지 경계 근접도, keypoint visibility,
  occlusion 추정 등으로 각 추정치의 신뢰도를 동적 조정

[한계]
  - 학습 기반 calibration (Learned Confidence Model)이 아닌 rule-based
  - 실제 GT 데이터 확보 후 학습 기반으로 교체 권장
"""
import numpy as np
from typing import Optional, Tuple

from utils.image_utils import bbox_area_ratio, bbox_edge_proximity


# ══════════════════════════════════════════════════════════════
#  공통 동적 신뢰도 인자
# ══════════════════════════════════════════════════════════════

def size_confidence(bbox: Tuple, img_h: int, img_w: int) -> float:
    """
    bbox 크기 기반 신뢰도
    - 너무 작은 객체 → 거리가 멀어 추정 부정확
    - 기준: 이미지 면적의 1% 이상이면 신뢰도 최대
    """
    ratio = bbox_area_ratio(bbox, img_h, img_w)
    if ratio < 0.002:   return 0.35
    elif ratio < 0.01:  return 0.55
    elif ratio < 0.05:  return 0.80
    else:               return 1.00


def edge_confidence(bbox: Tuple, img_h: int, img_w: int) -> float:
    """
    이미지 경계 근접도 기반 신뢰도
    - 경계에 걸친 객체 → 부분만 보임 → 낮은 신뢰도
    """
    proximity = bbox_edge_proximity(bbox, img_h, img_w, margin=0.06)
    return max(1.0 - proximity * 0.6, 0.30)


def occlusion_estimate(bbox_h: int, full_h_estimate: int) -> float:
    """
    bbox 높이 / 예상 전체 높이 비율로 가림 추정
    - 비율이 낮을수록 많이 가려진 것 → 낮은 신뢰도
    """
    if full_h_estimate <= 0:
        return 0.5
    visible_ratio = min(bbox_h / full_h_estimate, 1.0)
    if visible_ratio > 0.85:  return 1.00
    elif visible_ratio > 0.60: return 0.75
    elif visible_ratio > 0.40: return 0.55
    else:                      return 0.35


# ══════════════════════════════════════════════════════════════
#  사람 전용 신뢰도 보정 (ChatGPT [2])
# ══════════════════════════════════════════════════════════════

def person_pose_confidence(keypoints_17x3: Optional[np.ndarray],
                            lower_kp_indices: list) -> float:
    """
    하체 keypoint visibility score 기반 신뢰도

    ChatGPT 제안: keypoint visibility confidence 활용
    - lower_kp_indices: ankle, knee, hip 인덱스
    - visibility score 낮으면 가려진 것 → 수위 계산 신뢰도 하락
    """
    if keypoints_17x3 is None:
        return 0.40

    scores = []
    for idx in lower_kp_indices:
        if idx < len(keypoints_17x3):
            scores.append(float(keypoints_17x3[idx, 2]))

    if not scores:
        return 0.40

    avg_vis = float(np.mean(scores))
    # visibility 평균 → 신뢰도 매핑
    if avg_vis > 0.8:   return 1.00
    elif avg_vis > 0.5: return 0.75
    elif avg_vis > 0.3: return 0.55
    else:               return 0.30


def person_posture_quality(keypoints_17x3: Optional[np.ndarray],
                            bbox_h: int) -> Tuple[str, float]:
    """
    자세 판별 (ChatGPT [2]: standing/sitting/crouching 분류)

    별도 분류 모델 없이 keypoint 기하학으로 추정:
    - hip_y vs knee_y 거리 비율로 standing/sitting 구분
    - 앉거나 웅크리면 수위 계산 오차 → 신뢰도 하락

    Returns: (posture, quality_factor)
    """
    if keypoints_17x3 is None or bbox_h < 30:
        return "unknown", 0.60

    def ky(idx):
        return float(keypoints_17x3[idx, 1]) if keypoints_17x3[idx, 2] > 0.3 else None

    hip_y    = _avg(ky(11), ky(12))
    knee_y   = _avg(ky(13), ky(14))
    ankle_y  = _avg(ky(15), ky(16))
    shoulder_y = _avg(ky(5), ky(6))

    if hip_y is None or shoulder_y is None:
        return "unknown", 0.60

    torso_h = abs(shoulder_y - hip_y)

    if knee_y is None:
        return "standing_partial", 0.65

    hip_knee_dist = abs(hip_y - knee_y)
    # 서 있는 경우: hip~knee 거리가 torso보다 큰 편
    if hip_knee_dist > torso_h * 0.6:
        posture = "standing"
        quality = 0.90
    elif hip_knee_dist > torso_h * 0.3:
        posture = "crouching"
        quality = 0.55   # 수위 계산 신뢰도 하락
    else:
        posture = "sitting"
        quality = 0.35   # 앉은 경우 수위 계산 매우 부정확

    return posture, quality


def _avg(*vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


# ══════════════════════════════════════════════════════════════
#  차량 전용 신뢰도 보정 (ChatGPT [3])
# ══════════════════════════════════════════════════════════════

def car_orientation_from_bbox(bbox: Tuple, img_h: int, img_w: int) -> Tuple[str, float]:
    """
    차량 orientation 추정 (ChatGPT [3])

    별도 모델 없이 bbox 비율로 front/rear/side 추정:
    - 가로 >> 세로: side view (측면)
    - 가로 ≈ 세로: front/rear view
    - 이미지 좌우 위치: 추가 보조

    Returns: (orientation, confidence)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1
    aspect = bw / max(bh, 1)

    if aspect > 1.8:
        orientation = "side"
        conf = 0.75
    elif aspect > 1.2:
        orientation = "side_angled"
        conf = 0.55
    else:
        orientation = "front_rear"
        conf = 0.60

    return orientation, conf


def car_visibility_confidence(seg_result,
                               car_bbox: Tuple,
                               img_h: int) -> float:
    """
    차량 부품 감지 비율 기반 신뢰도
    감지된 부품 수가 많을수록 신뢰도 높음
    """
    if seg_result is None or not hasattr(seg_result, "boxes"):
        return 0.25
    n_parts = len(seg_result.boxes)
    if n_parts >= 5:  return 0.90
    elif n_parts >= 3: return 0.70
    elif n_parts >= 1: return 0.50
    else:              return 0.25


# ══════════════════════════════════════════════════════════════
#  통합 신뢰도 계산
# ══════════════════════════════════════════════════════════════

def compute_final_confidence(base_conf: float,
                              size_conf: float,
                              edge_conf: float,
                              extra_conf: float = 1.0) -> float:
    """
    최종 신뢰도 = 기본 × 크기 × 경계 × 추가인자
    모두 0~1 범위, 곱셈으로 결합 (모두 높아야 최종 높음)
    """
    final = base_conf * size_conf * edge_conf * extra_conf
    return float(np.clip(final, 0.05, 1.0))
