"""
기하학 모듈: 지면 평면 + 소실점 (v5.0)

GPT 지적 [1][7] 수정:
  - Ground Plane은 "보조 신호"로만 사용, 독립 scale source가 있으면 우선
  - scale 물리적 상한 clamp (0.01 ~ 10.0 cm/px)
  - Homography 실패 시 fallback scale을 도로폭 기준으로 보수적으로 설정
"""
import cv2
import numpy as np
from typing import Optional
import dataclasses

from utils.dataclasses import GroundPlane, VanishingPoint

SCALE_MIN = 0.01   # 매우 넓게 찍힌 장면
SCALE_MAX = 10.0   # 매우 가까이 찍힌 장면


def detect_vanishing_point(img_bgr: np.ndarray) -> VanishingPoint:
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    roi   = gray[h // 2:, :]
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=60, minLineLength=80, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return VanishingPoint(valid=False)

    filtered = []
    for ln in lines:
        x1,y1,x2,y2 = ln[0]
        dx,dy = x2-x1, y2-y1
        if abs(dx) < 1e-3:
            continue
        if 0.05 < abs(dy/dx) < 10:
            filtered.append(ln[0])

    if len(filtered) < 4:
        return VanishingPoint(valid=False)

    intersections = []
    for i in range(len(filtered)):
        for j in range(i+1, len(filtered)):
            pt = _line_intersect(filtered[i], filtered[j], h//2)
            if pt:
                px,py = pt
                if 0 <= px <= w and h*0.2 <= py <= h*0.9:
                    intersections.append([px,py])

    if len(intersections) < 3:
        return VanishingPoint(valid=False)

    pts  = np.array(intersections)
    vp_x = float(np.median(pts[:,0]))
    vp_y = float(np.median(pts[:,1]))
    return VanishingPoint(valid=True, x=vp_x, y=vp_y, horizon_y=vp_y)


def _line_intersect(l1, l2, y_offset):
    x1,y1,x2,y2 = l1; x3,y3,x4,y4 = l2
    y1+=y_offset; y2+=y_offset; y3+=y_offset; y4+=y_offset
    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    t  = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
    return x1+t*(x2-x1), y1+t*(y2-y1)


def estimate_ground_plane(img_h, img_w,
                           vp: Optional[VanishingPoint]=None,
                           flood_mask=None) -> GroundPlane:
    try:
        horizon_y  = vp.y if (vp and vp.valid) else img_h * 0.40
        horizon_y  = max(horizon_y, img_h * 0.20)
        w_bot = img_w * 0.88
        w_top = img_w * (0.30 if not (vp and vp.valid) else 0.25)
        offset = ((vp.x - img_w/2) * 0.3) if (vp and vp.valid) else 0.0

        src_pts = np.float32([
            [img_w/2 - w_top/2 + offset, horizon_y],
            [img_w/2 + w_top/2 + offset, horizon_y],
            [img_w/2 + w_bot/2,           img_h-1  ],
            [img_w/2 - w_bot/2,           img_h-1  ],
        ])
        dst_w, dst_h = img_w*0.6, img_h*0.6
        dst_pts = np.float32([
            [img_w/2-dst_w/2, img_h/2-dst_h/2],
            [img_w/2+dst_w/2, img_h/2-dst_h/2],
            [img_w/2+dst_w/2, img_h/2+dst_h/2],
            [img_w/2-dst_w/2, img_h/2+dst_h/2],
        ])
        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            return GroundPlane(valid=False)

        # 도로 폭 3m 기준, clamp 적용
        initial_scale = float(np.clip(300.0 / w_bot, SCALE_MIN, SCALE_MAX))
        return GroundPlane(valid=True, H=H,
                           scale_cm_per_px=initial_scale,
                           horizon_y=float(horizon_y),
                           vanishing_pt=vp, calibrated=False)
    except Exception:
        return GroundPlane(valid=False)


def refine_with_object(gp: GroundPlane,
                        pixel_height: float,
                        real_height_cm: float,
                        source: str="") -> GroundPlane:
    if pixel_height <= 0:
        return gp
    refined = float(np.clip(real_height_cm / pixel_height, SCALE_MIN, SCALE_MAX))
    return dataclasses.replace(gp, scale_cm_per_px=refined,
                                calibrated=True, calibration_src=source)


def pixels_to_cm(pixels: float, gp: Optional[GroundPlane]) -> float:
    if gp and gp.valid and gp.scale_cm_per_px > 0:
        return float(np.clip(pixels * gp.scale_cm_per_px, 0, 1000))
    return 0.0