"""
기하학 모듈: 지면 평면 추정 + 소실점 검출
(ChatGPT 피드백 [1][6] 반영 — geometry 독립 모듈화)

한계 명시:
  - 카메라 내부 파라미터(intrinsic) 없이는 절대 metric 보장 불가
  - 객체 실제 크기(사람 키, 차량 높이)로 scale 보정 필수
  - 경사 도로 / 심한 tilt 에서 오차 증가 가능
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import dataclasses

from utils.dataclasses import GroundPlane, VanishingPoint


# ══════════════════════════════════════════════════════════════
#  소실점 추정
# ══════════════════════════════════════════════════════════════

def detect_vanishing_point(img_bgr: np.ndarray) -> VanishingPoint:
    """
    엣지 기반 소실점 추정

    방법: Canny → HoughLinesP → 선 교점 클러스터링
    도로 장면에서 차선/건물 경계선의 소실점 = 지평선 위치

    한계: 직선 패턴이 뚜렷하지 않으면 (야간, 폭우) 실패 가능
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape

    # 하단 절반만 사용 (도로 영역)
    roi   = gray[h // 2:, :]
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=60, minLineLength=80, maxLineGap=20
    )
    if lines is None or len(lines) < 4:
        return VanishingPoint(valid=False)

    # 수평선 제거 (slope ≈ 0), 너무 가파른 선 제거
    filtered = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1e-3:
            continue
        slope = abs(dy / dx)
        if 0.05 < slope < 10:
            filtered.append(ln[0])

    if len(filtered) < 4:
        return VanishingPoint(valid=False)

    # 쌍별 교점 계산 (ROI 오프셋 보정)
    intersections = []
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            pt = _line_intersect(filtered[i], filtered[j], h // 2)
            if pt is not None:
                px, py = pt
                if 0 <= px <= w and h * 0.2 <= py <= h * 0.9:
                    intersections.append([px, py])

    if len(intersections) < 3:
        return VanishingPoint(valid=False)

    pts   = np.array(intersections)
    vp_x  = float(np.median(pts[:, 0]))
    vp_y  = float(np.median(pts[:, 1]))

    return VanishingPoint(
        valid=True,
        x=vp_x,
        y=vp_y,
        horizon_y=vp_y,
    )


def _line_intersect(l1, l2, y_offset: int):
    """두 선분의 교점 반환 (실패 시 None)"""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    y1 += y_offset; y2 += y_offset
    y3 += y_offset; y4 += y_offset

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return px, py


# ══════════════════════════════════════════════════════════════
#  지면 평면 추정 (Homography)
# ══════════════════════════════════════════════════════════════

def estimate_ground_plane(img_h: int, img_w: int,
                           vp: Optional[VanishingPoint] = None,
                           flood_mask: Optional[np.ndarray] = None,
                           ) -> GroundPlane:
    """
    Heuristic Ground Plane Homography 추정

    소실점(vanishing point)이 있으면 더 정확한 사다리꼴 설정.
    없으면 이미지 비율 기반 기본값 사용.

    [한계]
    - 카메라 캘리브레이션 데이터 없음 → 절대 scale 부정확
    - 객체 기반 scale 보정(refine_with_object)으로 반드시 후처리 필요
    - 경사로, 카메라 tilt > 15° 시 오차 급증
    """
    try:
        # 소실점 기반 지평선 y좌표
        if vp and vp.valid:
            horizon_y = vp.y
        else:
            horizon_y = img_h * 0.40   # 기본값: 이미지 상단 40%

        ground_top_y = horizon_y
        ground_top_y = max(ground_top_y, img_h * 0.20)

        # 도로 사다리꼴 (src) → 직사각형 (dst)
        w_bot = img_w * 0.88
        if vp and vp.valid:
            # 소실점 x에 따라 좌우 비대칭 보정
            cx     = vp.x
            offset = (cx - img_w / 2) * 0.3
            w_top  = img_w * 0.30
        else:
            w_top  = img_w * 0.35
            offset = 0.0

        src_pts = np.float32([
            [img_w / 2 - w_top / 2 + offset, ground_top_y],
            [img_w / 2 + w_top / 2 + offset, ground_top_y],
            [img_w / 2 + w_bot / 2,          img_h - 1   ],
            [img_w / 2 - w_bot / 2,          img_h - 1   ],
        ])

        dst_w = img_w * 0.6
        dst_h = img_h * 0.6
        dst_pts = np.float32([
            [img_w / 2 - dst_w / 2, img_h / 2 - dst_h / 2],
            [img_w / 2 + dst_w / 2, img_h / 2 - dst_h / 2],
            [img_w / 2 + dst_w / 2, img_h / 2 + dst_h / 2],
            [img_w / 2 - dst_w / 2, img_h / 2 + dst_h / 2],
        ])

        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            return GroundPlane(valid=False)

        # 초기 scale 추정: 도로 폭 약 3m(300cm) 기준
        initial_scale = 300.0 / w_bot

        return GroundPlane(
            valid=True,
            H=H,
            scale_cm_per_px=initial_scale,
            horizon_y=float(horizon_y),
            vanishing_pt=vp,
            calibrated=False,
        )

    except Exception as e:
        return GroundPlane(valid=False)


def refine_with_object(gp: GroundPlane,
                        pixel_height: float,
                        real_height_cm: float,
                        source: str = "") -> GroundPlane:
    """
    알려진 객체 실제 높이로 scale 보정 (ChatGPT [1][2] 핵심)

    pixel_height: 객체 픽셀 높이 (예: 사람 head~foot)
    real_height_cm: 객체 실제 높이 (예: 171cm)
    """
    if pixel_height <= 0:
        return gp
    refined = real_height_cm / pixel_height
    return dataclasses.replace(
        gp,
        scale_cm_per_px=refined,
        calibrated=True,
        calibration_src=source,
    )


# ══════════════════════════════════════════════════════════════
#  픽셀 → cm 변환
# ══════════════════════════════════════════════════════════════

def pixels_to_cm(pixels: float, gp: Optional[GroundPlane]) -> float:
    """픽셀 거리 → cm 변환 (GroundPlane scale 사용)"""
    if gp and gp.valid and gp.scale_cm_per_px > 0:
        return pixels * gp.scale_cm_per_px
    return 0.0
