"""
시스템 전반 데이터 클래스 (v4.0)
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
    uncertainty_cm:  float           = 0.0   # 추정 불확실도 (±cm)


@dataclass
class PersonMeta:
    """사람 속성 분류 결과"""
    is_child:    bool  = False
    gender:      str   = "male"
    body_key:    str   = "adult_male"
    child_conf:  float = 0.5
    gender_conf: float = 0.5
    pose_quality: float = 0.0   # keypoint visibility 평균


@dataclass
class CarMeta:
    """차량 속성 분류 결과"""
    car_type:      str   = "sedan"   # bus/mini/pickup/sedan/suv
    car_type_conf: float = 0.5
    orientation:   str   = "unknown" # front/rear/side/unknown
    yolo_cls_id:   int   = 2


@dataclass
class VanishingPoint:
    """소실점 추정 결과"""
    valid:   bool  = False
    x:       float = 0.0
    y:       float = 0.0
    horizon_y: float = 0.0   # 지평선 y좌표


@dataclass
class GroundPlane:
    """지면 평면 추정 결과 (Homography 포함)"""
    valid:           bool              = False
    H:               Optional[np.ndarray] = None
    scale_cm_per_px: float             = 0.0
    horizon_y:       float             = 0.0
    vanishing_pt:    Optional[VanishingPoint] = None
    calibrated:      bool              = False   # 객체 기반 보정 여부
    calibration_src: str               = ""      # 보정에 사용된 소스


@dataclass
class WaterlineResult:
    """RANSAC 수면선 추정 결과"""
    valid:        bool  = False
    waterline_y:  float = 0.0
    slope:        float = 0.0
    flood_ratio:  float = 0.0
    pixel_ratio:  float = 0.0
    inlier_ratio: float = 0.0


@dataclass
class AnalysisResult:
    """전체 이미지 분석 최종 결과"""
    # 기본 정보
    image_path:         str   = ""
    is_flooded:         bool  = False
    flood_conf:         float = 0.0

    # 중간 결과
    flood_mask:         Optional[np.ndarray]   = None
    waterline:          Optional[WaterlineResult] = None
    ground_plane:       Optional[GroundPlane]  = None

    # 수위 추정
    depth_estimates:    List[DepthEstimate]    = field(default_factory=list)
    used_estimates:     List[DepthEstimate]    = field(default_factory=list)   # IQR 후
    outlier_estimates:  List[DepthEstimate]    = field(default_factory=list)   # 제거된 것

    # 최종 수위
    avg_depth_cm:       float = 0.0
    weighted_depth_cm:  float = 0.0
    uncertainty_cm:     float = 0.0   # 추정치 분산 기반 불확실도
    calibrated_weights: dict  = field(default_factory=dict)

    # 위험 단계
    level_label:        str   = "매우 낮음 💧"
    level_color:        str   = "#64B5F6"

    # 메타
    proc_time:          float = 0.0
    warnings:           List[str] = field(default_factory=list)