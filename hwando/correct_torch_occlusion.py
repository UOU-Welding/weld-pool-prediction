"""
TF030_FLIR 용접기 토치 occlusion 보정 스크립트
- 27~45초(frame 810~1350) 구간에서 토치가 비드를 가리는 영역을 보간
- Convex Hull Gap Detection + Grayscale Inpainting + Re-segmentation
"""

import cv2
import numpy as np
import os
import glob

# ============================================================
# 파라미터
# ============================================================
FRAME_START = 810
FRAME_END = 1350

# 컬러바 제외
COLORBAR_X_MIN = 710

# Occlusion 검출
MIN_CONTOUR_AREA = 100
MAX_BRIDGE_DIST = 80           # contour 간 최대 연결 거리(px)
NOTCH_RATIO_THRESH = 0.35      # 병목 판정: min_width / avg_width

# Inpainting
INPAINT_RADIUS = 10
INPAINT_DILATE_KERNEL = 5
INPAINT_METHOD = cv2.INPAINT_TELEA

# 재분할
BEAD_THRESH_PERCENTILE = 15
SMOOTH_KERNEL = 7

# 경로
BASE = "/data/KHD/workspace/2026용접/결과/260327"
FRAMES_DIR = f"{BASE}/TF030_FLIR.mp4_frames"
MASKS_DIR = f"{BASE}/TF030_FLIR.mp4_masks"
OUTPUT_MASKS_DIR = f"{BASE}/TF030_FLIR.mp4_masks_corrected"
OUTPUT_COMPARE_DIR = f"{BASE}/TF030_FLIR.mp4_occlusion_comparison"
# ============================================================


def exclude_colorbar(mask):
    """컬러바 영역(x >= COLORBAR_X_MIN)을 mask에서 제거"""
    cleaned = mask.copy()
    cleaned[:, COLORBAR_X_MIN:] = 0
    return cleaned


def get_contours(mask):
    """면적 기준으로 필터링된 contour 목록 반환"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]


def contour_min_distance(c1, c2):
    """두 contour 간 최소 거리 계산"""
    pts1 = c1.reshape(-1, 2).astype(np.float32)
    pts2 = c2.reshape(-1, 2).astype(np.float32)
    # 샘플링으로 속도 최적화
    if len(pts1) > 200:
        idx = np.linspace(0, len(pts1) - 1, 200, dtype=int)
        pts1 = pts1[idx]
    if len(pts2) > 200:
        idx = np.linspace(0, len(pts2) - 1, 200, dtype=int)
        pts2 = pts2[idx]
    diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    return dists.min()


def detect_occlusion_disconnected(mask, contours):
    """Case A: 2+ contour가 가까이 있으면 gap 영역을 occlusion으로 검출"""
    if len(contours) < 2:
        return None

    # contour 쌍 간 거리 계산, 가까운 것들을 같은 비드 그룹으로
    bead_group = [contours[0]]
    remaining = list(contours[1:])

    changed = True
    while changed and remaining:
        changed = False
        for i, c in enumerate(remaining):
            for gc in bead_group:
                if contour_min_distance(gc, c) < MAX_BRIDGE_DIST:
                    bead_group.append(c)
                    remaining.pop(i)
                    changed = True
                    break
            if changed:
                break

    if len(bead_group) < 2:
        return None

    # bead 그룹의 convex hull
    all_pts = np.vstack([c.reshape(-1, 2) for c in bead_group])
    hull = cv2.convexHull(all_pts)
    hull_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(hull_mask, hull, 255)

    # hull에서 기존 mask를 빼면 gap 영역
    occlusion_zone = cv2.subtract(hull_mask, mask)

    # 컬러바 영역 제외
    occlusion_zone[:, COLORBAR_X_MIN:] = 0

    if np.count_nonzero(occlusion_zone) < MIN_CONTOUR_AREA:
        return None

    return occlusion_zone


def detect_occlusion_notch(mask, contours):
    """Case B: 단일 contour에서 병목(notch) 구간 검출"""
    if len(contours) != 1:
        return None

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    if w < 20 or h < 10:
        return None

    # x축 방향 width profile
    roi = mask[y:y + h, x:x + w]
    width_profile = np.sum(roi > 0, axis=0)  # 각 열의 white 픽셀 수

    nonzero_widths = width_profile[width_profile > 0]
    if len(nonzero_widths) < 10:
        return None

    avg_width = np.mean(nonzero_widths)
    threshold_width = avg_width * NOTCH_RATIO_THRESH

    # 병목 구간 찾기: width < threshold인 연속 구간
    is_notch = width_profile < threshold_width
    # 양 끝 제외 (비드 시작/끝은 자연스럽게 좁음)
    margin = max(5, int(w * 0.05))
    is_notch[:margin] = False
    is_notch[-margin:] = False

    # 연속 구간 찾기
    notch_runs = []
    start = None
    for i in range(len(is_notch)):
        if is_notch[i] and start is None:
            start = i
        elif not is_notch[i] and start is not None:
            notch_runs.append((start, i))
            start = None
    if start is not None:
        notch_runs.append((start, len(is_notch)))

    if not notch_runs:
        return None

    # 가장 긴 병목 구간 선택
    longest = max(notch_runs, key=lambda r: r[1] - r[0])
    notch_x_start, notch_x_end = longest

    if notch_x_end - notch_x_start < 3:
        return None

    # 병목 좌우의 centroid_y와 width 측정 (여유있는 참조점)
    ref_margin = 10
    left_x = max(0, notch_x_start - ref_margin)
    right_x = min(len(width_profile) - 1, notch_x_end + ref_margin)

    def get_centroid_y(col_x):
        col = roi[:, col_x]
        ys = np.where(col > 0)[0]
        if len(ys) == 0:
            return h // 2, 0
        return np.mean(ys), len(ys)

    left_cy, left_w = get_centroid_y(left_x)
    right_cy, right_w = get_centroid_y(right_x)

    if left_w == 0 and right_w == 0:
        return None

    # 기대 비드 영역 생성 (선형 보간)
    occlusion_zone = np.zeros_like(mask)
    span = notch_x_end - notch_x_start
    for i in range(span):
        t = i / max(1, span - 1)
        cy = left_cy * (1 - t) + right_cy * t
        ew = left_w * (1 - t) + right_w * t

        col_x_abs = x + notch_x_start + i
        y_top = int(y + cy - ew / 2)
        y_bot = int(y + cy + ew / 2)
        y_top = max(0, y_top)
        y_bot = min(mask.shape[0], y_bot)

        occlusion_zone[y_top:y_bot, col_x_abs] = 255

    # 기존 mask 부분은 제외
    occlusion_zone = cv2.subtract(occlusion_zone, mask)

    if np.count_nonzero(occlusion_zone) < 30:
        return None

    return occlusion_zone


def detect_occlusion(mask):
    """Occlusion 영역 검출 (disconnected 또는 notch)"""
    cleaned = exclude_colorbar(mask)
    contours = get_contours(cleaned)

    if len(contours) == 0:
        return None, "NONE"

    # Case A: Disconnected
    zone = detect_occlusion_disconnected(cleaned, contours)
    if zone is not None:
        return zone, "DISCONN"

    # Case B: Notch
    zone = detect_occlusion_notch(cleaned, contours)
    if zone is not None:
        return zone, "NOTCH"

    return None, "NONE"


def inpaint_occlusion(gray, occlusion_zone):
    """Grayscale inpainting으로 가려진 영역 복원"""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (INPAINT_DILATE_KERNEL, INPAINT_DILATE_KERNEL)
    )
    dilated = cv2.dilate(occlusion_zone, kernel, iterations=1)
    inpainted = cv2.inpaint(gray, dilated, INPAINT_RADIUS, INPAINT_METHOD)
    return inpainted


def correct_mask(gray, mask, occlusion_zone, correction_type):
    """보정 mask 생성 - gap 직접 채우기 방식"""
    if correction_type == "DISCONN":
        # Disconnected: gap 영역을 직접 mask에 채워넣기
        # convex hull gap이 이미 올바른 영역이므로 그대로 합산
        corrected = cv2.bitwise_or(mask, occlusion_zone)
    else:
        # Notch: 보간된 기대 영역을 직접 채우기
        corrected = cv2.bitwise_or(mask, occlusion_zone)

    # morphological closing으로 경계 평활화
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SMOOTH_KERNEL, SMOOTH_KERNEL))
    corrected = cv2.morphologyEx(corrected, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 작은 노이즈 제거
    contours = get_contours(corrected)
    clean = np.zeros_like(corrected)
    for c in contours:
        cv2.drawContours(clean, [c], -1, 255, -1)
    # 컬러바 영역 원본 유지
    clean[:, COLORBAR_X_MIN:] = mask[:, COLORBAR_X_MIN:]

    return clean


def create_comparison(gray, orig_mask, corr_mask, occlusion_zone, frame_idx):
    """4패널 비교 시각화 생성"""
    h, w = gray.shape

    # Panel 1: 원본 + contour
    p1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cleaned_orig = exclude_colorbar(orig_mask)
    contours_orig = get_contours(cleaned_orig)
    cv2.drawContours(p1, contours_orig, -1, (0, 255, 0), 1)

    # Panel 2: 원본 mask
    p2 = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGR)

    # Panel 3: 보정 mask (복원 영역 초록색)
    p3 = cv2.cvtColor(corr_mask, cv2.COLOR_GRAY2BGR)
    if occlusion_zone is not None:
        diff = cv2.subtract(corr_mask, orig_mask)
        p3[diff > 0] = (0, 255, 0)

    # Panel 4: 보정 overlay
    p4 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = p4.copy()
    overlay[corr_mask > 0] = (0, 0, 255)
    p4 = cv2.addWeighted(overlay, 0.4, p4, 0.6, 0)

    # 라벨
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.4, 1
    cv2.putText(p1, f"frame_{frame_idx:04d} + contours", (5, 15), font, fs, (0, 255, 255), th)
    cv2.putText(p2, "Original mask", (5, 15), font, fs, (255, 255, 255), th)
    cv2.putText(p3, "Corrected (green=restored)", (5, 15), font, fs, (0, 255, 0), th)
    cv2.putText(p4, "Corrected overlay", (5, 15), font, fs, (0, 255, 255), th)

    # 구분선
    sep_v = np.ones((h, 2, 3), dtype=np.uint8) * 80
    sep_h = np.ones((2, w * 2 + 2, 3), dtype=np.uint8) * 80

    top = np.hstack([p1, sep_v, p2])
    bot = np.hstack([p3, sep_v, p4])
    result = np.vstack([top, sep_h, bot])

    return result


def validate_correction(orig_mask, corr_mask):
    """보정 결과 검증"""
    cleaned_orig = exclude_colorbar(orig_mask)
    cleaned_corr = exclude_colorbar(corr_mask)

    orig_contours = get_contours(cleaned_orig)
    corr_contours = get_contours(cleaned_corr)

    # width profile 기반 병목 비율
    notch_ratio = 1.0
    if corr_contours:
        cnt = max(corr_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10:
            roi = cleaned_corr[y:y + h, x:x + w]
            wp = np.sum(roi > 0, axis=0)
            nz = wp[wp > 0]
            if len(nz) > 5:
                notch_ratio = np.min(nz) / np.mean(nz)

    return {
        "orig_contours": len(orig_contours),
        "corr_contours": len(corr_contours),
        "notch_ratio": notch_ratio,
        "orig_area": np.count_nonzero(cleaned_orig),
        "corr_area": np.count_nonzero(cleaned_corr),
        "added_pixels": np.count_nonzero(cleaned_corr) - np.count_nonzero(cleaned_orig),
    }


def process_frame(frame_idx):
    """단일 프레임 처리"""
    fname = f"frame_{frame_idx:04d}.png"
    frame_path = os.path.join(FRAMES_DIR, fname)
    mask_path = os.path.join(MASKS_DIR, fname)

    if not os.path.exists(frame_path) or not os.path.exists(mask_path):
        return None

    frame_bgr = cv2.imread(frame_path)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # occlusion 검출
    occlusion_zone, correction_type = detect_occlusion(mask)

    if occlusion_zone is not None:
        corrected = correct_mask(gray, mask, occlusion_zone, correction_type)
    else:
        corrected = mask.copy()

    # 저장
    cv2.imwrite(os.path.join(OUTPUT_MASKS_DIR, fname), corrected)

    # 비교 시각화 (occlusion이 있는 프레임만)
    if occlusion_zone is not None:
        comparison = create_comparison(gray, mask, corrected, occlusion_zone, frame_idx)
        cv2.imwrite(os.path.join(OUTPUT_COMPARE_DIR, fname), comparison)

    # 검증
    stats = validate_correction(mask, corrected)
    stats["type"] = correction_type
    stats["frame"] = frame_idx

    return stats


def main():
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_COMPARE_DIR, exist_ok=True)

    total = FRAME_END - FRAME_START + 1
    print(f"TF030_FLIR 토치 occlusion 보정: frame {FRAME_START}~{FRAME_END} ({total}프레임)")
    print(f"파라미터: bridge_dist={MAX_BRIDGE_DIST}, notch_thresh={NOTCH_RATIO_THRESH}")
    print(f"         inpaint_radius={INPAINT_RADIUS}, bead_percentile={BEAD_THRESH_PERCENTILE}")
    print()

    results = {"DISCONN": [], "NOTCH": [], "NONE": []}

    for idx in range(FRAME_START, FRAME_END + 1):
        stats = process_frame(idx)
        if stats:
            results[stats["type"]].append(stats)

        done = idx - FRAME_START + 1
        if done % 100 == 0 or done == total:
            print(f"  {done}/{total} 완료")

    # 요약
    print(f"\n{'='*65}")
    print("보정 결과 요약")
    print(f"{'='*65}")
    print(f"  Disconnected 보정: {len(results['DISCONN'])}프레임")
    print(f"  Notch 보정:        {len(results['NOTCH'])}프레임")
    print(f"  보정 불필요:       {len(results['NONE'])}프레임")

    corrected_all = results["DISCONN"] + results["NOTCH"]
    if corrected_all:
        avg_added = np.mean([s["added_pixels"] for s in corrected_all])
        avg_notch = np.mean([s["notch_ratio"] for s in corrected_all])
        single_contour = sum(1 for s in corrected_all if s["corr_contours"] == 1)
        print(f"\n  평균 추가 픽셀: {avg_added:.0f}px")
        print(f"  평균 병목비율(보정후): {avg_notch:.3f}")
        print(f"  단일 contour 달성: {single_contour}/{len(corrected_all)}")

        print(f"\n  {'프레임':<12} {'타입':<10} {'원본cnt':<9} {'보정cnt':<9} "
              f"{'추가px':<9} {'병목비':<8}")
        print(f"  {'-'*57}")
        for s in corrected_all[:30]:
            print(f"  frame_{s['frame']:04d}  {s['type']:<10} {s['orig_contours']:<9} "
                  f"{s['corr_contours']:<9} {s['added_pixels']:<9} {s['notch_ratio']:.3f}")
        if len(corrected_all) > 30:
            print(f"  ... 외 {len(corrected_all) - 30}프레임")


if __name__ == "__main__":
    main()
