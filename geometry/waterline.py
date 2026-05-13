"""
기하학 모듈: RANSAC 기반 수면선 추출
(ChatGPT [4] 반영 — geometry 독립 모듈화)
"""
import cv2
import numpy as np
from typing import Optional, Tuple

from utils.dataclasses import WaterlineResult


def _ransac_line(points: np.ndarray,
                 n_iter: int = 150,
                 thresh: float = 8.0) -> Tuple[float, float, float]:
    """
    2D 점 집합에서 RANSAC 직선 피팅
    Returns: (mid_y, slope, inlier_ratio)
    """
    n = len(points)
    if n < 5:
        return float(np.median(points[:, 1])), 0.0, 0.3

    best_inliers   = np.array([], dtype=int)
    best_slope     = 0.0
    best_intercept = float(np.median(points[:, 1]))

    for _ in range(n_iter):
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        dx = p2[0] - p1[0]
        if abs(dx) < 1e-6:
            continue
        slope     = (p2[1] - p1[1]) / dx
        intercept = p1[1] - slope * p1[0]
        residuals = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        inliers   = np.where(residuals < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers   = inliers
            best_slope     = slope
            best_intercept = intercept

    if len(best_inliers) >= 3:
        in_pts = points[best_inliers]
        if (in_pts[:, 0].max() - in_pts[:, 0].min()) > 10:
            c = np.polyfit(in_pts[:, 0], in_pts[:, 1], 1)
            best_slope, best_intercept = c[0], c[1]

    mid_x = float(np.mean(points[:, 0]))
    mid_y = best_slope * mid_x + best_intercept
    return float(mid_y), float(best_slope), len(best_inliers) / n


def extract_waterline(flood_mask: np.ndarray,
                       img_h: int, img_w: int) -> WaterlineResult:
    """
    침수 마스크 → RANSAC 수면선 추출

    전처리:
      1) Morphological Opening  → 반사/웅덩이 소음 제거
      2) Closing               → 경계 연결
    추출:
      3) 열별 침수 마스크 상단 y좌표 → 수면 후보점
      4) RANSAC 직선 피팅
    """
    if flood_mask is None or flood_mask.sum() == 0:
        return WaterlineResult(valid=False)

    mask_h, mask_w = flood_mask.shape[:2]
    binary  = (flood_mask > 0.5).astype(np.uint8)

    # 형태학적 노이즈 제거
    k       = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k)

    # 수면 후보점 추출 (하단 70% ROI)
    roi_start = int(mask_h * 0.30)
    roi       = cleaned[roi_start:, :]

    pts = []
    for col in range(0, mask_w, 4):
        col_d = roi[:, col]
        if col_d.sum() == 0:
            continue
        row = int(np.argmax(col_d)) + roi_start
        pts.append([col, row])

    if len(pts) < 5:
        return WaterlineResult(valid=False)

    pts_arr = np.array(pts, dtype=np.float32)
    wl_y_mask, slope, inlier_ratio = _ransac_line(pts_arr)

    # mask 좌표 → 원본 이미지 좌표
    wl_y_img  = wl_y_mask * (img_h / mask_h)
    flood_r   = max(0.0, min((img_h - wl_y_img) / img_h, 1.0))
    pixel_r   = float(cleaned.sum()) / max(cleaned.size, 1)

    return WaterlineResult(
        valid=True,
        waterline_y=float(wl_y_img),
        slope=float(slope),
        flood_ratio=float(flood_r),
        pixel_ratio=float(pixel_r),
        inlier_ratio=float(inlier_ratio),
    )
