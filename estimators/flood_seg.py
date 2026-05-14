"""
침수 Seg 기반 수위 추정기 (v5.2 - 최종)

핵심 설계 원칙:
  Seg는 "얼마나 잠겼는가(비율)" 를 알고,
  객체 추정기는 "참조 높이(cm)" 를 안다.
  → Seg flood_ratio × 참조 실제높이 = 수위

왜 scale_cm_per_px를 쓰면 안 되나:
  scale = 사람키(cm) / 사람bbox픽셀
  → 사람이 멀리 있으면 scale이 폭발적으로 커짐
  → submerged_px × scale = 수백 cm 폭주
  → 원근 왜곡이 있는 CCTV에서 근본적으로 부정확

올바른 계산:
  flood_ratio = (img_h - waterline_y) / img_h
  참조높이 = 검출된 객체의 실제높이(cm) 평균
  depth_cm = flood_ratio × 참조높이 × 보정계수

  예: flood_ratio=0.35, 차량높이 147cm → 35% × 147cm = 51.5cm
  예: flood_ratio=0.20, 성인 어깨높이 141cm → 20% × 141cm = 28.2cm

보정계수 (perspective_factor):
  화면 상단 수면 = 실제 거리가 멀어 비율이 과대평가될 수 있음
  → 0.6 ~ 0.8 범위로 보수적 보정
"""
import numpy as np
from typing import Optional, List

from utils.dataclasses import DepthEstimate, WaterlineResult

# 원근 보정 계수 (heuristic, 보수적 적용)
PERSPECTIVE_FACTOR = 0.70   # 수면이 화면 위쪽일수록 실제보다 과대평가
# Seg 신뢰도 상한 (보조 소스)
SEG_MAX_CONF = 0.55
# 참조 높이가 없을 때 사용할 기본값 (성인 어깨 높이)
DEFAULT_REF_HEIGHT_CM = 141.5  # 한국 성인 남성 어깨 높이


def estimate_depth_seg(
    waterline:       WaterlineResult,
    img_h:           int,
    img_w:           int,
    scale_hint:      Optional[float] = None,   # 더 이상 사용 안 함 (설계 변경)
    ground_y:        Optional[float] = None,
    ref_heights_cm:  Optional[List[float]] = None,  # 검출 객체들의 실제높이(cm) 목록
) -> Optional[DepthEstimate]:
    """
    flood_ratio × 참조 높이(cm) 기반 수위 추정

    Args:
        waterline:      extract_waterline() 결과
        img_h/w:        원본 이미지 크기
        scale_hint:     (무시됨, 하위호환용)
        ground_y:       지면 y좌표 (미사용, 하위호환용)
        ref_heights_cm: 검출된 객체들의 실제높이(cm) 리스트
                        (사람 → 어깨높이, 차량 → 전체높이)
    """
    if not waterline.valid:
        return None

    flood_ratio = waterline.flood_ratio   # 이미지 하단 기준 수면 비율 (0~1)

    if flood_ratio <= 0.01:
        return None

    # ── 참조 높이 결정 ────────────────────────────────────────
    if ref_heights_cm and len(ref_heights_cm) > 0:
        # 검출된 객체 중앙값 사용 (이상값 제거)
        ref_h = float(np.median(ref_heights_cm))
        ref_source = f"객체{len(ref_heights_cm)}개중앙값"
    else:
        ref_h = DEFAULT_REF_HEIGHT_CM
        ref_source = "기본값(성인어깨)"

    # ── 수위 계산 ─────────────────────────────────────────────
    # flood_ratio × 참조높이 × 원근보정
    # 수면이 화면 위쪽(flood_ratio 클수록)일수록 보정 강화
    # (멀리서 찍힌 수면은 실제보다 과대평가됨)
    if flood_ratio > 0.5:
        factor = PERSPECTIVE_FACTOR * 0.85   # 화면 위쪽 수면 → 더 강한 보정
    elif flood_ratio > 0.3:
        factor = PERSPECTIVE_FACTOR
    else:
        factor = min(PERSPECTIVE_FACTOR * 1.1, 0.90)  # 낮은 수면 → 덜 보정

    depth_cm    = float(np.clip(flood_ratio * ref_h * factor, 0.0, 250.0))
    uncertainty = depth_cm * 0.30   # flood_ratio 자체 불확실도 + 원근 보정 오차

    # ── 신뢰도 ────────────────────────────────────────────────
    # RANSAC 품질 × 픽셀 커버리지
    # 참조 높이가 실제 객체에서 왔으면 신뢰도 상승
    base_conf = waterline.inlier_ratio * 0.6 + waterline.pixel_ratio * 0.4
    ref_bonus  = 0.10 if (ref_heights_cm and len(ref_heights_cm) > 0) else 0.0
    conf = float(np.clip(base_conf + ref_bonus, 0.0, SEG_MAX_CONF))

    return DepthEstimate(
        source="flood_seg",
        depth_cm=depth_cm,
        confidence=conf,
        detail=(
            f"boundary y={waterline.waterline_y:.0f}px | "
            f"flood_ratio:{flood_ratio:.1%} | "
            f"ref_h:{ref_h:.0f}cm({ref_source}) | "
            f"factor:{factor:.2f} | "
            f"inlier:{waterline.inlier_ratio:.0%}"
        ),
        bbox=None,
        scale_cm_per_px=None,
        method="boundary_ratio",
        uncertainty_cm=float(uncertainty),
    )