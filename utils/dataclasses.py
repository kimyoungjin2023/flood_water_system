"""
시스템 전반 데이터 클래스 (v5.1)
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class DepthEstimate:
    """단일 객체에서 추정된 수위"""
    source:          str
    depth_cm:        float
    confidence:      float
    detail:          str             = ""
    bbox:            Optional[Tuple] = None
    scale_cm_per_px: Optional[float] = None
    method:          str             = "unknown"
    uncertainty_cm:  float           = 0.0


@dataclass
class PersonMeta:
    is_child:     bool  = False
    gender:       str   = "male"
    body_key:     str   = "adult_male"
    child_conf:   float = 0.5
    gender_conf:  float = 0.5
    pose_quality: float = 0.0


@dataclass
class CarMeta:
    car_type:      str   = "sedan"
    car_type_conf: float = 0.5
    orientation:   str   = "unknown"
    yolo_cls_id:   int   = 2


@dataclass
class VanishingPoint:
    valid:     bool  = False
    x:         float = 0.0
    y:         float = 0.0
    horizon_y: float = 0.0


@dataclass
class GroundPlane:
    valid:           bool              = False
    H:               Optional[np.ndarray] = None
    scale_cm_per_px: float             = 0.0
    horizon_y:       float             = 0.0
    vanishing_pt:    Optional[VanishingPoint] = None
    calibrated:      bool              = False
    calibration_src: str               = ""


@dataclass
class WaterlineResult:
    """boundary 기반 수면선 추정 결과 (v5.1)"""
    valid:        bool  = False
    waterline_y:  float = 0.0    # 이미지 내 수면 y좌표 (px)
    slope:        float = 0.0    # 수면선 기울기
    flood_ratio:  float = 0.0    # (img_h - waterline_y) / img_h
    pixel_ratio:  float = 0.0    # 침수 픽셀 비율
    inlier_ratio: float = 0.0    # RANSAC inlier 비율
    # v5.1 추가
    boundary_width_ratio: float = 0.0  # 수평 연속 경계선 너비 비율
    method:       str   = "boundary"   # 검출 방법


@dataclass
class AnalysisResult:
    """전체 이미지 분석 최종 결과"""
    image_path:         str   = ""
    is_flooded:         bool  = False
    flood_conf:         float = 0.0

    flood_mask:         Optional[np.ndarray]    = None
    waterline:          Optional[WaterlineResult] = None
    ground_plane:       Optional[GroundPlane]   = None

    depth_estimates:    List[DepthEstimate]     = field(default_factory=list)
    used_estimates:     List[DepthEstimate]     = field(default_factory=list)
    outlier_estimates:  List[DepthEstimate]     = field(default_factory=list)

    avg_depth_cm:       float = 0.0
    weighted_depth_cm:  float = 0.0
    uncertainty_cm:     float = 0.0
    calibrated_weights: dict  = field(default_factory=dict)

    level_label:        str   = "매우 낮음 💧"
    level_color:        str   = "#64B5F6"

    proc_time:          float = 0.0
    warnings:           List[str] = field(default_factory=list)