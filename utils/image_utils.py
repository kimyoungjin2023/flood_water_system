"""
이미지 전처리 공통 유틸리티
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Tuple, List

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def to_tensor(img_bgr: np.ndarray,
              size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """BGR numpy → ImageNet 정규화 텐서"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tf  = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(_MEAN, _STD)])
    return tf(pil).unsqueeze(0)


def softmax_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).detach().cpu().numpy()[0]


def avg_valid(*vals) -> Optional[float]:
    """None 제외 평균"""
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def safe_crop(img: np.ndarray,
              x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((4, 4, 3), np.uint8)


def bbox_area_ratio(bbox: Tuple, img_h: int, img_w: int) -> float:
    """bbox 면적 / 이미지 면적 비율"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1) / max(img_h * img_w, 1)


def bbox_edge_proximity(bbox: Tuple,
                         img_h: int, img_w: int,
                         margin: float = 0.05) -> float:
    """
    bbox가 이미지 경계에 가까운 비율 반환 (0=안전, 1=경계에 걸침)
    Dynamic Confidence 계산에 사용 (ChatGPT [5])
    """
    x1, y1, x2, y2 = bbox
    proximity = max(
        margin - x1 / img_w,
        margin - y1 / img_h,
        (x2 / img_w) - (1 - margin),
        (y2 / img_h) - (1 - margin),
        0.0,
    )
    return min(proximity / margin, 1.0)