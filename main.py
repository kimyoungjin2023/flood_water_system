"""
╔══════════════════════════════════════════════════════════════════╗
║   침수 감지 및 수위 측정 통합 시스템  v4.0                      ║
║   Vision-based Multi-modal Flood Depth Estimation System         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  [v4.0 적용된 개선사항]                                          ║
║                                                                  ║
║  ✅ [1] 소실점(Vanishing Point) 검출 + Ground Plane Homography   ║
║         geometry/ground_plane.py 독립 모듈화                     ║
║                                                                  ║
║  ✅ [2] 사람 Pose visibility confidence 동적 보정                ║
║         자세(서있음/앉음/웅크림) 판별 → 신뢰도 반영             ║
║                                                                  ║
║  ✅ [3] EfficientNet-B0 차량 종류 분류 (5cls)                   ║
║         bus / car-mini / car-pickup / car-sedan / car-suv        ║
║         Vehicle orientation(front/rear/side) 추정               ║
║                                                                  ║
║  ✅ [4] RANSAC 수면선 + Morphological 노이즈 제거               ║
║         geometry/waterline.py 독립 모듈화                        ║
║                                                                  ║
║  ✅ [5] Dynamic Confidence Calibration                           ║
║         bbox 크기 / 경계 근접도 / keypoint visibility /          ║
║         불확실도(uncertainty) 기반 가중치 자동 조정              ║
║                                                                  ║
║  ✅ [6] 완전 모듈화 구조                                        ║
║         geometry/ confidence/ estimators/ fusion/                ║
║         visualization/ models/ utils/                            ║
║                                                                  ║
║  ✅ [7] 조건부 inference skip                                   ║
║         사람 없으면 Pose skip, 차량 없으면 Car Seg skip          ║
║                                                                  ║
║  ✅ [추가] 불확실도(uncertainty) 출력                           ║
║         각 추정치 ±cm 표시 + 시각화 오차막대                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

프로젝트 구조:
  flood_system/
  ├── utils/          config.py  dataclasses.py  image_utils.py
  ├── models/         loader.py
  ├── geometry/       ground_plane.py  waterline.py
  ├── confidence/     calibration.py
  ├── estimators/     person.py  car.py  infrastructure.py  flood_seg.py
  ├── fusion/         weighted_fusion.py
  └── visualization/  visualizer.py
"""

import sys, os, time, warnings, logging

# ── 환경 설정 ─────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import cv2
import numpy as np
from pathlib import Path

# ── 내부 모듈 ─────────────────────────────────────────────────
from utils.config      import OUTPUT_DIR, MODEL_DIR, YOLO_CLS_TO_CAR
from utils.dataclasses import AnalysisResult
from utils.image_utils  import to_tensor, softmax_np, safe_crop

from models.loader     import ModelLoader

from geometry.ground_plane import (
    detect_vanishing_point, estimate_ground_plane, refine_with_object,
)
from geometry.waterline    import extract_waterline

from estimators.person        import estimate_depth_person
from estimators.car           import estimate_depth_car
from estimators.infrastructure import (
    estimate_depth_sign, estimate_depth_tubular,
)
from estimators.flood_seg     import estimate_depth_seg

from fusion.weighted_fusion   import fuse, get_flood_level
from visualization.visualizer import visualize


# ══════════════════════════════════════════════════════════════
#  메인 파이프라인
# ══════════════════════════════════════════════════════════════

class FloodDetectionSystem:
    def __init__(self):
        self.loader = ModelLoader()
        self.M      = self.loader.M

    # ── 공개 API ─────────────────────────────────────────────
    def analyze(self, image_path: str,
                output_dir: str = OUTPUT_DIR) -> AnalysisResult:
        t0 = time.time()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"이미지 읽기 실패: {image_path}")
        img_h, img_w = img.shape[:2]
        stem = Path(image_path).stem

        print(f"\n{'━'*64}")
        print(f"  분석: {Path(image_path).name}  ({img_w}×{img_h}px)")
        print(f"{'━'*64}")

        result = AnalysisResult(image_path=image_path)

        # ─────────────────────────────────────────────────────
        # STEP 1: 침수 상황 분류
        # ─────────────────────────────────────────────────────
        print("\n[STEP 1] 침수 상황 분류 (ResNet-18)...")
        result.is_flooded, result.flood_conf = self._classify_situation(img)
        print(f"  → {'침수 ⚠️ ' if result.is_flooded else '정상 ✅'}  "
              f"(신뢰도: {result.flood_conf:.1%})")

        # ─────────────────────────────────────────────────────
        # STEP 2: Semantic Segmentation + RANSAC 수면선
        # ─────────────────────────────────────────────────────
        print("\n[STEP 2] 침수 Seg + RANSAC 수면선 추출...")
        result.flood_mask = self._run_seg(img)
        if result.flood_mask is not None:
            pct = result.flood_mask.mean()
            print(f"  → 침수 픽셀 비율: {pct:.2%}")
            result.waterline = extract_waterline(result.flood_mask, img_h, img_w)
            if result.waterline.valid:
                wl = result.waterline
                print(f"  → RANSAC 수면선 y={wl.waterline_y:.0f}px | "
                      f"inlier:{wl.inlier_ratio:.0%} | "
                      f"flood_ratio:{wl.flood_ratio:.1%}")
            else:
                print("  → 수면선 추출 실패 (침수 픽셀 부족)")
        else:
            print("  → Seg 모델 없음, 건너뜀")

        # ─────────────────────────────────────────────────────
        # STEP 3: 소실점 검출 + Ground Plane Homography
        # ─────────────────────────────────────────────────────
        print("\n[STEP 3] 소실점 검출 + Ground Plane 추정...")
        vp = detect_vanishing_point(img)
        if vp.valid:
            print(f"  → 소실점: ({vp.x:.0f}, {vp.y:.0f})  "
                  f"지평선 y={vp.horizon_y:.0f}px")
        else:
            print("  → 소실점 검출 실패 (기본값 사용)")

        result.ground_plane = estimate_ground_plane(img_h, img_w, vp, result.flood_mask)
        if result.ground_plane.valid:
            gp = result.ground_plane
            print(f"  → Homography 추정 완료 | "
                  f"초기 scale≈{gp.scale_cm_per_px:.2f}cm/px | "
                  f"horizon_y={gp.horizon_y:.0f}px")
        else:
            print("  → Homography 추정 실패")

        # ─────────────────────────────────────────────────────
        # STEP 4: 객체 탐지 (사람 / 차량)
        # ─────────────────────────────────────────────────────
        print("\n[STEP 4] 객체 탐지 (사람 / 차량)...")
        persons, cars = self._detect_objects(img)
        print(f"  → 사람: {len(persons)}명 | 차량: {len(cars)}대")

        # ─────────────────────────────────────────────────────
        # STEP 5: 사람 분석 [조건부 — 사람 없으면 skip]
        # ─────────────────────────────────────────────────────
        if persons:
            print("\n[STEP 5] 사람 분석 (아동/성인, 성별, Pose, 자세판별)...")
            pose_kps = self._run_pose(img)   # Pose는 사람 있을 때만

            for i, (bbox, cls_id) in enumerate(persons):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = safe_crop(img, x1, y1, x2, y2)
                kps  = pose_kps[i] if i < len(pose_kps) else None

                est = estimate_depth_person(
                    crop=crop,
                    bbox=bbox,
                    kps_17x3=kps,
                    img_h=img_h,
                    img_w=img_w,
                    child_model=self.M.get("child"),
                    gender_model=self.M.get("gender"),
                    ground_plane=result.ground_plane,
                )
                if est is None:
                    continue

                # 사람 키 scale로 GroundPlane 보정
                if (est.scale_cm_per_px
                        and result.ground_plane
                        and result.ground_plane.valid
                        and not result.ground_plane.calibrated):
                    result.ground_plane = refine_with_object(
                        result.ground_plane,
                        pixel_height=float(y2 - y1),
                        real_height_cm=est.scale_cm_per_px * (y2 - y1),
                        source=f"person[{i+1}]",
                    )
                    print(f"  → GroundPlane scale 보정: "
                          f"{result.ground_plane.scale_cm_per_px:.3f}cm/px "
                          f"(사람[{i+1}] 기준)")

                result.depth_estimates.append(est)
                print(f"  → 사람[{i+1}] {est.detail[:65]} | "
                      f"수위: {est.depth_cm:.1f}cm ±{est.uncertainty_cm:.1f}cm")
        else:
            print("\n[STEP 5] 사람 없음 — Pose estimation skip ✂️")

        # ─────────────────────────────────────────────────────
        # STEP 6: 차량 분석 [조건부 — 차량 없으면 skip]
        # ─────────────────────────────────────────────────────
        if cars:
            print("\n[STEP 6] 차량 분석 (EfficientNet 종류 분류 + 부품 Seg)...")
            for i, (bbox, cls_id) in enumerate(cars):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop    = safe_crop(img, x1, y1, x2, y2)
                car_seg = self._run_car_seg(crop)   # 차량 있을 때만

                est = estimate_depth_car(
                    car_bbox=bbox,
                    yolo_cls_id=cls_id,
                    seg_result=car_seg,
                    img_h=img_h,
                    img_w=img_w,
                    car_cls_model=self.M.get("car_cls"),
                    crop=crop,
                    ground_plane=result.ground_plane,
                )
                result.depth_estimates.append(est)
                print(f"  → 차량[{i+1}] {est.detail} | "
                      f"수위: {est.depth_cm:.1f}cm ±{est.uncertainty_cm:.1f}cm")
        else:
            print("\n[STEP 6] 차량 없음 — Car Segmentation skip ✂️")

        # ─────────────────────────────────────────────────────
        # STEP 7: 표지판 탐지
        # ─────────────────────────────────────────────────────
        print("\n[STEP 7] 표지판 탐지 (기준: 3.2m)...")
        signs = self._detect(img, "yolo_sign")
        print(f"  → 표지판: {len(signs)}개")
        for bbox in signs:
            est = estimate_depth_sign(bbox, img_h, img_w)
            result.depth_estimates.append(est)
            print(f"  → {est.detail}")

        # ─────────────────────────────────────────────────────
        # STEP 8: 시선유도봉 탐지
        # ─────────────────────────────────────────────────────
        print("\n[STEP 8] 시선유도봉 탐지 (기준: 80cm)...")
        tubes = self._detect(img, "yolo_tube")
        print(f"  → 시선유도봉: {len(tubes)}개")
        for bbox in tubes:
            est = estimate_depth_tubular(bbox, img_h, img_w)
            result.depth_estimates.append(est)
            print(f"  → {est.detail}")

        # ─────────────────────────────────────────────────────
        # STEP 9: Flood Seg 수위 추정 (scale hint 적용)
        # ─────────────────────────────────────────────────────
        if result.waterline and result.waterline.valid:
            print("\n[STEP 9] Seg 수면선 → 수위 변환...")
            # 가장 신뢰도 높은 scale hint 선택
            scale_hint = self._best_scale_hint(result)
            seg_est = estimate_depth_seg(
                result.waterline, img_h, img_w, scale_hint)
            if seg_est:
                result.depth_estimates.append(seg_est)
                print(f"  → Seg 수위: {seg_est.depth_cm:.1f}cm ±{seg_est.uncertainty_cm:.1f}cm "
                      f"(신뢰도: {seg_est.confidence:.0%})")

        # ─────────────────────────────────────────────────────
        # STEP 10: Weighted Fusion (IQR + Uncertainty-aware)
        # ─────────────────────────────────────────────────────
        print("\n[STEP 10] Weighted Fusion (IQR outlier 제거 + 불확실도 보정)...")
        avg, weighted, uncertainty, w_dict, used, outliers = \
            fuse(result.depth_estimates)

        result.avg_depth_cm       = avg
        result.weighted_depth_cm  = weighted
        result.uncertainty_cm     = uncertainty
        result.calibrated_weights = w_dict
        result.used_estimates     = used
        result.outlier_estimates  = outliers

        if outliers:
            print(f"  → IQR 제거: "
                  f"{[f'{e.source}:{e.depth_cm:.1f}cm' for e in outliers]}")

        result.level_label, result.level_color = get_flood_level(weighted)
        result.proc_time = time.time() - t0

        self._print_report(result)

        # ─────────────────────────────────────────────────────
        # STEP 11: 시각화 저장
        # ─────────────────────────────────────────────────────
        out_path = f"{output_dir}/{stem}_flood_v4.jpg"
        visualize(img, result, out_path)

        return result

    # ── 내부 헬퍼 메서드 ─────────────────────────────────────

    def _classify_situation(self, img: np.ndarray):
        m = self.M.get("situation")
        if m is None:
            return False, 0.5
        try:
            import torch
            with torch.no_grad():
                p = softmax_np(m(to_tensor(img)))
            return bool(p[1] > p[0]), float(max(p))
        except Exception as e:
            print(f"  [WARN] 분류 오류: {e}")
            return False, 0.5

    def _run_seg(self, img: np.ndarray):
        m = self.M.get("flood_seg")
        if m is None:
            return None
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = (cv2.resize(rgb, (256, 256)) / 255.0).astype(np.float32)[None]
            return m.predict(inp, verbose=0)[0, :, :, 0]
        except Exception as e:
            print(f"  [WARN] Seg 오류: {e}")
            return None

    def _detect_objects(self, img: np.ndarray):
        """YOLO로 사람/차량 탐지 → [(bbox, cls_id), ...] 반환"""
        m = self.M.get("yolo_main")
        if m is None:
            return [], []
        try:
            res = m(img, verbose=False, conf=0.3)[0]
            persons, cars = [], []
            for box in res.boxes:
                cls  = int(box.cls[0])
                xyxy = box.xyxy[0].tolist()
                if cls == 0:
                    persons.append((xyxy, cls))
                elif cls in YOLO_CLS_TO_CAR:
                    cars.append((xyxy, cls))
            return persons, cars
        except Exception as e:
            print(f"  [WARN] 객체 탐지 오류: {e}")
            return [], []

    def _run_pose(self, img: np.ndarray) -> list:
        """Pose 추정 — 사람이 있을 때만 호출 (조건부 skip)"""
        m = self.M.get("yolo_pose")
        if m is None:
            return []
        try:
            res = m(img, verbose=False, conf=0.3)[0]
            if res.keypoints is None:
                return []
            return [kp.cpu().numpy() for kp in res.keypoints.data]
        except Exception as e:
            print(f"  [WARN] Pose 오류: {e}")
            return []

    def _run_car_seg(self, crop: np.ndarray):
        """차량 부품 Seg — 차량이 있을 때만 호출 (조건부 skip)"""
        m = self.M.get("yolo_car")
        if m is None or crop.size == 0:
            return None
        try:
            return m(crop, verbose=False, conf=0.25)[0]
        except Exception as e:
            print(f"  [WARN] 차량 Seg 오류: {e}")
            return None

    def _detect(self, img: np.ndarray, key: str) -> list:
        m = self.M.get(key)
        if m is None:
            return []
        try:
            res = m(img, verbose=False, conf=0.3)[0]
            return [b.xyxy[0].tolist() for b in res.boxes]
        except Exception as e:
            print(f"  [WARN] {key} 탐지 오류: {e}")
            return []

    def _best_scale_hint(self, result: AnalysisResult):
        """
        다른 소스에서 calibration된 가장 신뢰도 높은 scale 반환
        우선순위: GroundPlane(calibrated) > person scale > car scale
        """
        # GroundPlane이 객체로 보정된 경우
        gp = result.ground_plane
        if gp and gp.valid and gp.calibrated and gp.scale_cm_per_px > 0:
            return gp.scale_cm_per_px

        # 사람 추정치에서 scale 추출 (가장 신뢰도 높은 것)
        person_scales = [
            e.scale_cm_per_px for e in result.depth_estimates
            if e.source == "person" and e.scale_cm_per_px
        ]
        if person_scales:
            return float(np.median(person_scales))

        # Homography 기본 scale
        if gp and gp.valid and gp.scale_cm_per_px > 0:
            return gp.scale_cm_per_px

        return None

    def _print_report(self, result: AnalysisResult):
        sep = "═" * 64
        print(f"\n╔{sep}╗")
        print(f"║{'최  종  분  석  결  과   v4.0':^64}║")
        print(f"╠{sep}╣")
        flood_str = '침수 감지됨 ⚠️ ' if result.is_flooded else '정상 ✅        '
        print(f"║  침수 여부  : {flood_str} (신뢰도 {result.flood_conf:.1%}){' '*18}║")
        print(f"║  단순 평균  : {result.avg_depth_cm:6.1f} cm{' '*48}║")
        print(f"║  가중 평균  : {result.weighted_depth_cm:6.1f} cm  ◀  최종 수위{' '*36}║")
        print(f"║  불확실도   : ± {result.uncertainty_cm:5.1f} cm{' '*46}║")
        print(f"║  위험 단계  : {result.level_label}{' '*50}║")
        print(f"║  처리 시간  : {result.proc_time:.2f}초{' '*50}║")
        print(f"╠{sep}╣")
        print(f"║  사용된 추정치 ({len(result.used_estimates)}개 / 전체 {len(result.depth_estimates)}개){' '*38}║")
        print(f"║  {'소스':<10} {'수위':>8} {'신뢰도':>7} {'±cm':>6}  방법{' '*28}║")
        print(f"╠{'─'*64}╣")
        for e in result.used_estimates:
            icon = {"person":"👤","car":"🚗","sign":"🚧",
                    "tubular":"🏮","flood_seg":"🗺️"}.get(e.source, "•")
            print(f"║  {icon} {e.source:<8} {e.depth_cm:>8.1f} {e.confidence:>7.0%}"
                  f" {e.uncertainty_cm:>5.1f}cm  {e.method:<20}{' '*12}║")
        if result.outlier_estimates:
            print(f"╠{'─'*64}╣")
            print(f"║  IQR 제거 ({len(result.outlier_estimates)}개): "
                  + ", ".join(f"{e.source}:{e.depth_cm:.1f}cm"
                              for e in result.outlier_estimates)
                  + " " * 20 + "║")
        print(f"╚{sep}╝")


# ══════════════════════════════════════════════════════════════
#  CLI 진입점
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="침수 감지 & 수위 측정 시스템 v4.0",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "images", nargs="*",
        help="분석할 이미지 경로 (생략 시 업로드 이미지 자동 사용)",
    )
    ap.add_argument("--output", default=OUTPUT_DIR, help="결과 저장 폴더")
    args = ap.parse_args()

    # 분석 대상 이미지 결정
    if args.images:
        targets = [p for p in args.images if Path(p).exists()]
        missing = [p for p in args.images if not Path(p).exists()]
        if missing:
            print(f"⚠️  파일 없음: {missing}")
    else:
        # 업로드 이미지 자동 탐색
        upload_dir = Path(MODEL_DIR).parent / "uploads"
        candidates = sorted(upload_dir.glob("*.png")) + sorted(upload_dir.glob("*.jpg"))
        # 표 이미지(신체치수, 차량치수)가 아닌 실제 장면 이미지만
        targets = [str(p) for p in candidates
                   if p.stat().st_size > 50_000]  # 50KB 이상만 (큰 이미지)

        if not targets:
            print("⚠️  분석할 이미지 없음 → 테스트 이미지 생성")
            dummy_path = "/tmp/test_flood_scene.jpg"
            # 도로 + 침수 더미 이미지 생성
            dummy = np.zeros((640, 960, 3), dtype=np.uint8)
            # 하늘 (상단)
            dummy[:200, :] = [135, 190, 210]
            # 건물/배경 (중단)
            dummy[200:350, :] = [80, 80, 90]
            # 도로 + 침수 (하단)
            dummy[350:, :, 0] = 40
            dummy[350:, :, 1] = 90
            dummy[350:, :, 2] = 160
            # 도로 경계선 (소실점 검출용 직선)
            cv2.line(dummy, (480, 300), (200, 640), (180, 180, 180), 3)
            cv2.line(dummy, (480, 300), (760, 640), (180, 180, 180), 3)
            cv2.imwrite(dummy_path, dummy)
            targets = [dummy_path]

    if not targets:
        print("❌ 분석할 이미지가 없습니다.")
        sys.exit(1)

    # 시스템 초기화 (모델 로드)
    system = FloodDetectionSystem()

    # 이미지 순차 분석
    all_results = []
    for img_path in targets:
        print(f"\n\n{'#'*66}")
        print(f"#  {img_path}")
        print(f"{'#'*66}")
        try:
            r = system.analyze(img_path, args.output)
            all_results.append(r)
        except Exception as e:
            import traceback
            print(f"❌ 분석 실패: {e}")
            traceback.print_exc()

    # 다중 이미지 처리 시 전체 요약
    if len(all_results) > 1:
        print(f"\n\n{'═'*64}")
        print(f"  전체 분석 완료: {len(all_results)}개 이미지")
        print(f"{'═'*64}")
        for r in all_results:
            name = Path(r.image_path).name
            print(f"  {name:<40} "
                  f"{r.weighted_depth_cm:6.1f}cm ±{r.uncertainty_cm:.1f}cm  "
                  f"[{r.level_label}]")
        print(f"{'═'*64}")
