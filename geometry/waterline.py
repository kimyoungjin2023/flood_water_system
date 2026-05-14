"""
수면선(Waterline) 검출 모듈 (v5.1)

핵심 설계 변경:
  기존: mask 최상단 flood pixel → 수면선 (반사/그림자에 매우 취약)
  변경: flood ↔ non-flood 경계(boundary) 검출 → 수면선

왜 boundary 기반이어야 하나?
  실제 수면은 "물 있는 영역"의 최상단이 아니라
  "물 없는 영역과 물 있는 영역의 경계"임.
  mask 최상단은 반사광·그림자·노이즈가 포함되어 폭주함.
  경계선은 실제 물 표면에 가장 가까운 신호.

파이프라인:
  1. Morphology (노이즈 제거)
  2. Connected Component 필터 (작은 반사 덩어리 제거)
  3. ROI 제한 (이미지 하단 60% — 도로 영역만)
  4. Horizontal Density Projection (수평 연속성 검사)
     → 일정 너비 이상 연속된 행만 유효 (점 노이즈 제거)
  5. flood↔non-flood boundary contour 추출
  6. 가장 긴 수평 contour 선택
  7. RANSAC line fitting
  8. EMA temporal smoothing (직전 결과와 가중 평균)
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from utils.dataclasses import WaterlineResult

# ── 전역 EMA 상태 (단일 이미지 시스템에서도 연속 호출 시 안정화) ──
_ema_waterline_y: Optional[float] = None
_EMA_ALPHA = 0.35   # 새 프레임 반영 비율 (0.35 = 65%는 이전 유지)

# ── 파라미터 ──────────────────────────────────────────────────
# ROI: 이미지 하단 몇 %를 도로 영역으로 간주
ROI_TOP_RATIO   = 0.35   # 상단 35% 제거 (하늘·건물·반사 제거)
# 수평 연속성: 전체 너비의 몇 % 이상 연속돼야 유효한 수면선인가
MIN_WIDTH_RATIO = 0.25   # 25% 이상 연속 → 유효
# CC 필터: 전체 마스크 면적의 몇 % 이상이어야 유효 컴포넌트인가
MIN_CC_AREA_RATIO = 0.04
# RANSAC
RANSAC_ITER   = 200
RANSAC_THRESH = 6.0      # 픽셀 단위 inlier 거리
MIN_INLIER    = 0.40     # inlier 비율 최소치
# 기울기 제한: 수면선은 거의 수평이어야 함
MAX_SLOPE_DEG = 20.0     # ±20도 이상 기울면 무효


def reset_ema():
    """새 영상 시작 시 EMA 상태 초기화"""
    global _ema_waterline_y
    _ema_waterline_y = None


def _ransac_line(points: np.ndarray) -> Tuple[float, float, float]:
    """
    RANSAC 직선 피팅
    Returns: (mid_y, slope, inlier_ratio)
    """
    n = len(points)
    if n < 5:
        return float(np.median(points[:, 1])), 0.0, 0.3

    best_inliers = np.array([], dtype=int)
    best_slope, best_intercept = 0.0, float(np.median(points[:, 1]))

    for _ in range(RANSAC_ITER):
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        dx = p2[0] - p1[0]
        if abs(dx) < 1e-6:
            continue
        slope     = (p2[1] - p1[1]) / dx
        intercept = p1[1] - slope * p1[0]
        residuals = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        inliers   = np.where(residuals < RANSAC_THRESH)[0]
        if len(inliers) > len(best_inliers):
            best_inliers, best_slope, best_intercept = inliers, slope, intercept

    if len(best_inliers) >= 3:
        in_pts = points[best_inliers]
        if (in_pts[:, 0].max() - in_pts[:, 0].min()) > 10:
            c = np.polyfit(in_pts[:, 0], in_pts[:, 1], 1)
            best_slope, best_intercept = c[0], c[1]

    mid_x = float(np.mean(points[:, 0]))
    mid_y = best_slope * mid_x + best_intercept
    return float(mid_y), float(best_slope), len(best_inliers) / n


def _horizontal_density(binary: np.ndarray, mask_w: int) -> np.ndarray:
    """
    각 행(row)의 flood 픽셀 비율 계산
    → 수평으로 연속된 flood 행을 찾기 위해 사용
    """
    return binary.sum(axis=1) / max(mask_w, 1)


def _find_longest_boundary_contour(boundary_pts: np.ndarray,
                                    mask_w: int) -> Optional[np.ndarray]:
    """
    boundary 점들 중 수평으로 가장 넓게 퍼진 구간 반환
    → 점 노이즈·작은 반사 덩어리 제거
    """
    if len(boundary_pts) == 0:
        return None

    min_width = mask_w * MIN_WIDTH_RATIO
    # x 좌표 기준으로 연속 구간 탐색
    sorted_pts = boundary_pts[boundary_pts[:, 0].argsort()]
    best_pts, cur_pts = [], [sorted_pts[0]]

    for pt in sorted_pts[1:]:
        if pt[0] - cur_pts[-1][0] <= 8:   # x 간격 8px 이하면 연속
            cur_pts.append(pt)
        else:
            if len(cur_pts) > 0:
                xs = np.array([p[0] for p in cur_pts])
                if xs.max() - xs.min() >= min_width and len(cur_pts) > len(best_pts):
                    best_pts = cur_pts
            cur_pts = [pt]

    if cur_pts:
        xs = np.array([p[0] for p in cur_pts])
        if xs.max() - xs.min() >= min_width and len(cur_pts) > len(best_pts):
            best_pts = cur_pts

    return np.array(best_pts) if len(best_pts) >= 5 else None


def extract_waterline(flood_mask: np.ndarray,
                       img_h: int, img_w: int,
                       use_ema: bool = True) -> WaterlineResult:
    """
    flood↔non-flood boundary 기반 수면선 추출

    Args:
        flood_mask: (256,256) float32, 0~1
        img_h, img_w: 원본 이미지 크기
        use_ema: True면 직전 결과와 EMA 평균 (영상 연속 처리 시 유용)
    """
    global _ema_waterline_y

    if flood_mask is None or flood_mask.sum() == 0:
        return WaterlineResult(valid=False)

    mask_h, mask_w = flood_mask.shape[:2]

    # ── STEP 1: Threshold + Morphology ───────────────────────
    binary  = (flood_mask > 0.50).astype(np.uint8)

    # Opening: 반사·점 노이즈 제거
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # Closing: 경계 연결
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    # ── STEP 2: Connected Component 필터링 ────────────────────
    # 작은 반사 덩어리 제거 (전체 면적의 MIN_CC_AREA_RATIO 미만 제거)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8)
    if n_labels > 1:
        min_area = max(mask_h * mask_w * MIN_CC_AREA_RATIO, 80)
        filtered = np.zeros_like(cleaned)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == lbl] = 1
        if filtered.sum() > 0:
            cleaned = filtered

    # ── STEP 3: ROI 제한 (도로 영역만) ───────────────────────
    # 상단 ROI_TOP_RATIO 제거 → 하늘·건물·반사광 영역 차단
    roi_start = int(mask_h * ROI_TOP_RATIO)
    cleaned[:roi_start, :] = 0

    if cleaned.sum() == 0:
        return WaterlineResult(valid=False)

    # ── STEP 4: Horizontal Density Projection ─────────────────
    # 각 행의 flood 픽셀 비율 계산
    row_density = _horizontal_density(cleaned, mask_w)

    # 수평으로 연속적인 행만 유효 (MIN_WIDTH_RATIO 이상)
    valid_rows = np.where(row_density >= MIN_WIDTH_RATIO)[0]
    if len(valid_rows) == 0:
        return WaterlineResult(valid=False)

    # ── STEP 5: Boundary Contour 추출 ─────────────────────────
    # flood↔non-flood 경계 = boundary의 상단 contour가 수면선
    # contour 추출 후 각 x 좌표에서 가장 위의 경계 y좌표 수집
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return WaterlineResult(valid=False)

    # 모든 contour에서 각 x별 최소 y (= 수면선 상단 경계)
    boundary_dict = {}
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            if y >= roi_start:  # ROI 안에 있는 점만
                if x not in boundary_dict or y < boundary_dict[x]:
                    boundary_dict[x] = y

    if len(boundary_dict) < 10:
        return WaterlineResult(valid=False)

    boundary_pts = np.array([[x, y] for x, y in sorted(boundary_dict.items())],
                             dtype=np.float32)

    # ── STEP 6: 수평 연속성 검사 ─────────────────────────────
    valid_boundary = _find_longest_boundary_contour(boundary_pts, mask_w)
    if valid_boundary is None:
        # 연속 구간 없으면 전체 boundary 사용 (완화)
        if len(boundary_pts) < 5:
            return WaterlineResult(valid=False)
        valid_boundary = boundary_pts

    # ── STEP 7: RANSAC 직선 피팅 ─────────────────────────────
    wl_y_mask, slope, inlier_ratio = _ransac_line(valid_boundary)

    if inlier_ratio < MIN_INLIER:
        return WaterlineResult(valid=False)

    # 기울기 각도 검사 (수면은 거의 수평)
    slope_deg = abs(np.degrees(np.arctan(slope)))
    if slope_deg > MAX_SLOPE_DEG:
        # 기울기가 너무 크면 수평 강제 (slope=0, y=중앙값)
        wl_y_mask = float(np.median(valid_boundary[:, 1]))
        slope     = 0.0
        inlier_ratio *= 0.6   # 신뢰도 하락

    # ── STEP 8: mask → 원본 이미지 좌표 변환 ─────────────────
    scale_y     = img_h / mask_h
    wl_y_img    = wl_y_mask * scale_y

    # ── STEP 9: EMA Temporal Smoothing ───────────────────────
    # 영상처럼 연속 호출될 때 spike 억제
    # 단일 이미지면 그냥 그 값 사용 (EMA 초기화 상태)
    if use_ema and _ema_waterline_y is not None:
        prev_y   = _ema_waterline_y
        # 갑작스러운 spike 검사 (±img_h*20% 이상이면 부분만 반영)
        diff     = abs(wl_y_img - prev_y)
        if diff > img_h * 0.20:
            alpha = _EMA_ALPHA * 0.3   # spike면 새 값 반영 최소화
        else:
            alpha = _EMA_ALPHA
        wl_y_img = (1 - alpha) * prev_y + alpha * wl_y_img

    _ema_waterline_y = wl_y_img

    # ── 최종 메트릭 계산 ─────────────────────────────────────
    wl_y_img    = float(np.clip(wl_y_img, roi_start * scale_y, img_h - 1))
    flood_ratio = float(np.clip((img_h - wl_y_img) / img_h, 0.0, 1.0))
    pixel_ratio = float(cleaned.sum()) / max(cleaned.size, 1)

    return WaterlineResult(
        valid=True,
        waterline_y=wl_y_img,
        slope=float(slope),
        flood_ratio=flood_ratio,
        pixel_ratio=pixel_ratio,
        inlier_ratio=float(inlier_ratio),
    )