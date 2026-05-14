"""
시각화 모듈 (v5.0)

GPT 지적 [5] 수정:
  - NanumGothic 한글 폰트 자동 설정 (Linux/Windows 분기)
  - matplotlib 경고 없이 한글 렌더링
  - 패널 단순화: "최종 수위 + 근거 + 신뢰도" 위주
  - 상대 침수율(%) 게이지 추가
  - 불확실도 오차막대 유지
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
import platform
import os

from utils.config import OVERLAY_COLORS, OVERLAY_ICONS, FLOOD_LEVELS
from utils.dataclasses import AnalysisResult


# ── 한글 폰트 설정 (GPT [5]) ─────────────────────────────────
def _setup_korean_font():
    system = platform.system()
    candidates = []

    if system == "Windows":
        candidates = ["Malgun Gothic", "맑은 고딕", "NanumGothic"]
    else:  # Linux / Mac
        candidates = ["NanumGothic", "NanumBarunGothic", "Nanum Gothic",
                      "DejaVu Sans"]

    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name

    # 직접 파일로 로드 시도
    nanum_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in nanum_paths:
        if os.path.exists(p):
            prop = fm.FontProperties(fname=p)
            matplotlib.rcParams["font.family"] = prop.get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            return prop.get_name()

    matplotlib.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


FONT_NAME = _setup_korean_font()

BG, PANEL, TC, GRID, SPINE = "#111827","#1F2937","#E5E7EB","#374151","#4B5563"


def _ax(ax, bg=PANEL):
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)


def draw_overlay(img_bgr, result):
    ov = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    for est in result.used_estimates:
        if est.bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in est.bbox]
        c   = OVERLAY_COLORS.get(est.source, (200,200,200))
        cv2.rectangle(ov, (x1,y1),(x2,y2), c, 2)
        lbl = f"{est.source} {est.depth_cm:.0f}cm(+/-{est.uncertainty_cm:.0f})"
        bw  = len(lbl)*7+4
        ly  = max(y1-20, 0)
        cv2.rectangle(ov, (x1,ly),(x1+bw,y1), c, -1)
        cv2.putText(ov, lbl, (x1+2,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (15,15,15), 1, cv2.LINE_AA)
    if result.waterline and result.waterline.valid:
        wy = int(result.waterline.waterline_y)
        cv2.line(ov, (0,wy),(ov.shape[1],wy),(0,200,255),2)
        cv2.putText(ov, "waterline",(5,max(wy-5,15)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),1,cv2.LINE_AA)
    return ov


def draw_seg_overlay(img_bgr, result):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if result.flood_mask is None:
        return img_rgb
    vis  = img_rgb.copy().astype(np.float32)
    mask = cv2.resize(result.flood_mask.astype(np.float32),(w,h))
    vis[...,0] = np.clip(vis[...,0]-mask*70,  0,255)
    vis[...,2] = np.clip(vis[...,2]+mask*170, 0,255)
    res = vis.astype(np.uint8)
    ctrs,_ = cv2.findContours((mask>0.45).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, ctrs, -1, (0,200,200), 1)
    if result.waterline and result.waterline.valid:
        wy = int(result.waterline.waterline_y)
        cv2.line(res,(0,wy),(w,wy),(0,255,200),2)
    return res


def _draw_gauge(ax, depth_cm, uncertainty_cm, level_color):
    max_d = 250.0
    level_segs = [(0,10,"#64B5F6"),(10,30,"#4CAF50"),(30,50,"#FDD835"),
                  (50,80,"#FF9800"),(80,120,"#F44336"),(120,250,"#9C27B0")]
    ax.barh([0],[max_d],height=0.55,color="#374151",edgecolor=SPINE)
    for lo,hi,c in level_segs:
        w=min(hi,depth_cm)-lo
        if w>0: ax.barh([0],[w],left=lo,height=0.55,color=c,alpha=0.88)
    # 불확실도 범위 표시
    lo_u = max(0, depth_cm - uncertainty_cm)
    hi_u = min(max_d, depth_cm + uncertainty_cm)
    ax.barh([0],[hi_u-lo_u],left=lo_u,height=0.55,color="white",alpha=0.20)
    for mark,lbl in [(0,"0"),(30,"주의"),(50,"경고"),(80,"위험"),(120,"매우위험")]:
        ax.axvline(mark,color="#6B7280",lw=0.8,alpha=0.6)
        ax.text(mark,-0.38,lbl,ha="center",va="top",color="#9CA3AF",fontsize=7)
    tx = min(depth_cm+4, max_d*0.82)
    ax.text(tx, 0, f"  {depth_cm:.1f}cm\n  ±{uncertainty_cm:.1f}cm",
            va="center",color="white",fontsize=10,fontweight="bold",linespacing=1.3)
    ax.set_xlim(0,max_d); ax.set_ylim(-0.7,0.7); ax.set_yticks([])
    ax.set_xlabel("수위 (cm)",color=TC,fontsize=9); ax.tick_params(colors=TC)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)


def _draw_summary(ax, result):
    lines = [
        f"{'─'*100}",
        f"  침수 여부: {'침수 감지됨 ⚠' if result.is_flooded else '정상 ✅'}  "
        f"  분류기 신뢰도: {result.flood_conf:.1%}",
        f"",
        f"  수위 결과  │  단순 평균: {result.avg_depth_cm:6.1f}cm  "
        f"│  가중 평균: {result.weighted_depth_cm:6.1f}cm  "
        f"│  불확실도: ±{result.uncertainty_cm:.1f}cm  "
        f"│  위험 단계: {result.level_label}",
        f"",
        f"  {'─'*96}",
        f"  {'소스':<12} {'수위cm':>7} {'신뢰도':>7} {'불확실도':>9}  {'방법':<22}  세부",
        f"  {'─'*96}",
    ]
    for e in result.used_estimates:
        icon = OVERLAY_ICONS.get(e.source,"•")
        lines.append(
            f"  {icon} {e.source:<10} {e.depth_cm:>7.1f} {e.confidence:>7.0%}"
            f" ±{e.uncertainty_cm:>6.1f}cm  {e.method:<22}  {e.detail[:50]}"
        )
    if result.outlier_estimates:
        lines.append(f"\n  제거됨 ({len(result.outlier_estimates)}개): "
                     + ", ".join(f"{e.source}:{e.depth_cm:.1f}cm"
                                 for e in result.outlier_estimates))
    if result.ground_plane and result.ground_plane.valid:
        gp = result.ground_plane
        lines.append(f"\n  Homography | scale:{gp.scale_cm_per_px:.3f}cm/px | "
                     f"calibrated:{gp.calibrated}({gp.calibration_src})")
    if result.waterline and result.waterline.valid:
        wl = result.waterline
        lines.append(f"  수면선 | RANSAC inlier:{wl.inlier_ratio:.0%} | "
                     f"flood:{wl.flood_ratio:.1%} | slope:{wl.slope:.4f}")
    if result.warnings:
        lines.append(f"\n  경고: " + " | ".join(result.warnings))
    lines.append(f"\n  처리 시간: {result.proc_time:.2f}초  |  폰트: {FONT_NAME}")

    ax.text(0.01,0.97,"\n".join(lines),transform=ax.transAxes,
            color=TC,fontsize=8.2,va="top",ha="left",fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6",fc="#0D1117",ec=GRID,alpha=0.92))


def visualize(img_bgr, result, out_path):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]

    fig = plt.figure(figsize=(24,17), facecolor=BG)
    gs  = GridSpec(3,3,figure=fig,hspace=0.42,wspace=0.32)

    # ① 원본
    ax1=fig.add_subplot(gs[0,0]); ax1.imshow(img_rgb); ax1.axis("off"); _ax(ax1)
    ax1.set_title("① 원본 이미지",color=TC,fontsize=11,fontweight="bold",pad=8)

    # ② 탐지 오버레이
    ax2=fig.add_subplot(gs[0,1])
    ax2.imshow(draw_overlay(img_bgr,result)); ax2.axis("off"); _ax(ax2)
    ax2.set_title("② 탐지 & 수위 오버레이",color=TC,fontsize=11,fontweight="bold",pad=8)

    # ③ Seg
    ax3=fig.add_subplot(gs[0,2])
    if result.flood_mask is not None:
        ax3.imshow(draw_seg_overlay(img_bgr,result))
        t3="③ 침수 Seg + RANSAC 수면선"
    else:
        ax3.imshow(img_rgb)
        ax3.text(W//2,H//2,"Seg 미사용",ha="center",va="center",
                 color="#6B7280",fontsize=12,bbox=dict(boxstyle="round",fc="#374151"))
        t3="③ Seg (미사용)"
    ax3.axis("off"); _ax(ax3)
    ax3.set_title(t3,color=TC,fontsize=11,fontweight="bold",pad=8)

    # ④ 소스별 바차트
    ax4=fig.add_subplot(gs[1,:2]); _ax(ax4)
    used=result.used_estimates
    if used:
        labels=[f"{OVERLAY_ICONS.get(e.source,'•')} {e.source}\n{e.depth_cm:.1f}cm"
                for e in used]
        depths=[e.depth_cm for e in used]
        confs =[e.confidence for e in used]
        errs  =[e.uncertainty_cm for e in used]
        bcols =[tuple(v/255 for v in OVERLAY_COLORS.get(e.source,(160,160,160))) for e in used]
        bars=ax4.bar(range(len(labels)),depths,color=bcols,
                     edgecolor=GRID,lw=0.8,alpha=0.88,zorder=3)
        ax4.errorbar(range(len(labels)),depths,yerr=errs,fmt="none",
                     ecolor="#F87171",elinewidth=1.8,capsize=5,zorder=4)
        ax4.set_xticks(range(len(labels))); ax4.set_xticklabels(labels,color=TC,fontsize=9)
        ax4.set_ylabel("수위 (cm)",color=TC,fontsize=10)
        ax4.set_title("④ 소스별 수위 + 불확실도 오차막대",color=TC,fontsize=11,fontweight="bold")
        ax4.tick_params(colors=TC); ax4.grid(axis="y",color=GRID,lw=0.6,zorder=0)
        for sp in ax4.spines.values(): sp.set_edgecolor(SPINE)
        mx = max(depths) if depths else 1
        for bar,conf in zip(bars,confs):
            ax4.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+mx*0.02,
                     f"{conf:.0%}",ha="center",va="bottom",color="#D1D5DB",fontsize=8)
        if result.avg_depth_cm>0:
            ax4.axhline(result.avg_depth_cm,color="#F87171",ls="--",lw=1.5,zorder=5,
                        label=f"단순평균 {result.avg_depth_cm:.1f}cm")
        if result.weighted_depth_cm>0:
            ax4.axhline(result.weighted_depth_cm,color="#34D399",ls="-",lw=2.2,zorder=6,
                        label=f"가중평균 {result.weighted_depth_cm:.1f}cm")
        ax4.legend(facecolor=GRID,edgecolor=SPINE,labelcolor=TC,fontsize=9)
    else:
        ax4.text(0.5,0.5,"탐지 객체 없음",ha="center",va="center",
                 color="#6B7280",fontsize=13,transform=ax4.transAxes)

    # ⑤ 수위 게이지
    ax5=fig.add_subplot(gs[1,2]); _ax(ax5)
    _draw_gauge(ax5,result.weighted_depth_cm,result.uncertainty_cm,result.level_color)
    ax5.set_title(f"⑤ 최종 수위 게이지",color=TC,fontsize=11,fontweight="bold")

    # ⑥ 요약
    ax6=fig.add_subplot(gs[2,:]); _ax(ax6); ax6.axis("off")
    _draw_summary(ax6,result)

    flood_tag="⚠ 침수 감지됨" if result.is_flooded else "✅ 정상"
    fig.suptitle(
        f"침수 감지 & 수위 측정 시스템 v5.0  |  {flood_tag}  |  "
        f"최종: {result.weighted_depth_cm:.1f}cm ±{result.uncertainty_cm:.1f}cm  [{result.level_label}]",
        color="#F9FAFB",fontsize=13,fontweight="bold",y=0.98,
        bbox=dict(boxstyle="round,pad=0.5",fc="#1E3A5F",ec="#3B82F6",alpha=0.95),
    )
    plt.savefig(out_path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close()
    print(f"  저장: {out_path}")