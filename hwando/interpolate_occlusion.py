"""
FLIR 열화상 mask occlusion 보간 스크립트
- 연속 프레임 간 mask 비교로 occlusion 감지
- 공간적 보간 (inpainting) + 시간적 보간 (temporal interpolation)
- 보간된 mask 및 비교 시각화 저장
"""

import cv2
import numpy as np
import os
import glob

# ============================================================
# 파라미터 설정
# ============================================================
# Occlusion 감지
AREA_DROP_RATIO = 0.08         # 이전 프레임 대비 mask 면적이 이 비율 이상 감소하면 occlusion
IOU_THRESHOLD = 0.90           # 이전 프레임과의 IoU가 이 값 미만이면 occlusion 후보
MIN_OCCLUDED_PIXELS = 150      # 최소 occluded 픽셀 수 (이하 무시)
TEMPORAL_WINDOW = 5            # 시간적 보간에 사용할 전/후 프레임 수
INPAINT_RADIUS = 7             # inpainting 반지름
INPAINT_METHOD = cv2.INPAINT_TELEA  # INPAINT_TELEA 또는 INPAINT_NS
DILATION_KERNEL = 5            # occluded 영역 확장 커널 (inpainting 경계 품질 향상)
# ============================================================

BASE_RESULT = "/data/KHD/workspace/2026용접/결과"

TARGETS = [
    {
        "name": "TF016_FLIR",
        "frames_dir":  f"{BASE_RESULT}/260325/TF016_FLIR.mp4_frames",
        "masks_dir":   f"{BASE_RESULT}/260325/TF016_FLIR.mp4_masks",
        "interp_dir":  f"{BASE_RESULT}/260325/TF016_FLIR.mp4_masks_interpolated",
        "compare_dir": f"{BASE_RESULT}/260325/TF016_FLIR.mp4_comparison",
    },
    {
        "name": "TF027_FLIR",
        "frames_dir":  f"{BASE_RESULT}/260327/TF027_FLIR.mp4_frames",
        "masks_dir":   f"{BASE_RESULT}/260327/TF027_FLIR.mp4_masks",
        "interp_dir":  f"{BASE_RESULT}/260327/TF027_FLIR.mp4_masks_interpolated",
        "compare_dir": f"{BASE_RESULT}/260327/TF027_FLIR.mp4_comparison",
    },
    {
        "name": "TF030_FLIR",
        "frames_dir":  f"{BASE_RESULT}/260327/TF030_FLIR.mp4_frames",
        "masks_dir":   f"{BASE_RESULT}/260327/TF030_FLIR.mp4_masks",
        "interp_dir":  f"{BASE_RESULT}/260327/TF030_FLIR.mp4_masks_interpolated",
        "compare_dir": f"{BASE_RESULT}/260327/TF030_FLIR.mp4_comparison",
    },
]


def compute_iou(mask_a, mask_b):
    """두 binary mask의 IoU 계산"""
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    if union == 0:
        return 1.0
    return intersection / union


def detect_occlusion(prev_mask, curr_mask):
    """이전 프레임과 현재 프레임의 mask를 비교하여 occluded 영역 반환"""
    prev_area = np.count_nonzero(prev_mask)
    curr_area = np.count_nonzero(curr_mask)

    if prev_area == 0:
        return None, False

    # 조건 1: 면적 감소율 체크
    area_drop = (prev_area - curr_area) / prev_area
    # 조건 2: IoU 체크
    iou = compute_iou(prev_mask, curr_mask)

    is_occluded = (area_drop > AREA_DROP_RATIO) or (iou < IOU_THRESHOLD and prev_area > MIN_OCCLUDED_PIXELS)

    if not is_occluded:
        return None, False

    # occluded 영역: 이전에 있었지만 현재 사라진 영역
    occluded_region = cv2.subtract(prev_mask, curr_mask)
    occluded_pixels = np.count_nonzero(occluded_region)

    if occluded_pixels < MIN_OCCLUDED_PIXELS:
        return None, False

    return occluded_region, True


def spatial_inpaint(frame_bgr, curr_mask, occluded_region):
    """공간적 보간: occluded 영역을 주변 열 분포 기반으로 inpainting"""
    # occluded 영역을 약간 확장하여 경계 품질 향상
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATION_KERNEL, DILATION_KERNEL))
    inpaint_mask = cv2.dilate(occluded_region, kernel, iterations=1)

    # grayscale에서 inpainting 수행
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    inpainted_gray = cv2.inpaint(gray, inpaint_mask, INPAINT_RADIUS, INPAINT_METHOD)

    return inpainted_gray


def temporal_interpolate_mask(masks_list, frame_indices, target_idx):
    """시간적 보간: 전/후 프레임의 mask를 가중 평균하여 보간"""
    h, w = masks_list[target_idx].shape
    accumulated = np.zeros((h, w), dtype=np.float64)
    weight_sum = 0.0

    for offset in range(-TEMPORAL_WINDOW, TEMPORAL_WINDOW + 1):
        if offset == 0:
            continue
        neighbor_idx = target_idx + offset
        if neighbor_idx < 0 or neighbor_idx >= len(masks_list):
            continue

        neighbor_mask = masks_list[neighbor_idx]
        if np.count_nonzero(neighbor_mask) == 0:
            continue

        # 거리에 반비례하는 가중치
        weight = 1.0 / abs(offset)
        accumulated += neighbor_mask.astype(np.float64) * weight
        weight_sum += weight

    if weight_sum == 0:
        return masks_list[target_idx]

    averaged = accumulated / weight_sum
    # threshold를 적용하여 binary mask 생성 (가중 평균이 128 이상이면 열 영역)
    temporal_mask = (averaged >= 128).astype(np.uint8) * 255
    return temporal_mask


def combine_interpolation(curr_mask, spatial_gray, temporal_mask, occluded_region):
    """공간적 + 시간적 보간 결과를 결합하여 최종 mask 생성"""
    # 공간적 보간 결과에서 occluded 영역의 intensity가 높으면 열 영역으로 판단
    # inpainted grayscale에서 Otsu threshold
    _, spatial_binary = cv2.threshold(spatial_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # occluded 영역에서만 보간 결과를 적용
    # 공간적 + 시간적 보간의 합집합 (occluded 영역 내에서)
    restored_spatial = cv2.bitwise_and(spatial_binary, occluded_region)
    restored_temporal = cv2.bitwise_and(temporal_mask, occluded_region)
    restored = cv2.bitwise_or(restored_spatial, restored_temporal)

    # 원본 mask + 복원된 영역
    result = cv2.bitwise_or(curr_mask, restored)
    return result


def create_comparison(orig_mask, interp_mask, frame_idx):
    """원본 mask vs 보간된 mask 나란히 비교 이미지 생성"""
    h, w = orig_mask.shape

    # 3채널로 변환
    orig_vis = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGR)
    interp_vis = cv2.cvtColor(interp_mask, cv2.COLOR_GRAY2BGR)

    # 차이 영역을 초록색으로 표시
    diff = cv2.subtract(interp_mask, orig_mask)
    interp_vis[diff > 0] = (0, 255, 0)  # 보간으로 추가된 영역: 초록색

    # 구분선
    separator = np.ones((h, 3, 3), dtype=np.uint8) * 128

    # 나란히 배치
    comparison = np.hstack([orig_vis, separator, interp_vis])

    # 라벨 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, f"Original (frame_{frame_idx:04d})", (10, 25),
                font, 0.6, (255, 255, 255), 1)
    cv2.putText(comparison, "Interpolated (green=restored)", (w + 13, 25),
                font, 0.6, (0, 255, 0), 1)

    return comparison


def process_target(target):
    """하나의 영상 데이터셋에 대해 occlusion 감지 및 보간 수행"""
    name = target["name"]
    frames_dir = target["frames_dir"]
    masks_dir = target["masks_dir"]
    interp_dir = target["interp_dir"]
    compare_dir = target["compare_dir"]

    os.makedirs(interp_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(masks_dir, "frame_*.png")))
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    total = len(mask_files)

    if total == 0:
        print(f"[SKIP] {name}: mask 없음")
        return []

    print(f"\n{'='*60}")
    print(f"[{name}] {total}프레임 occlusion 보간 시작")
    print(f"{'='*60}")

    # 1단계: 모든 mask 로드
    print("  mask 로딩 중...")
    all_masks = []
    for mf in mask_files:
        m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        all_masks.append(m)

    # 2단계: occlusion 감지
    print("  occlusion 감지 중...")
    occluded_frames = {}  # {frame_idx: occluded_region}

    for i in range(1, total):
        occluded_region, is_occluded = detect_occlusion(all_masks[i - 1], all_masks[i])
        if is_occluded and occluded_region is not None:
            occluded_frames[i] = occluded_region

    print(f"  감지된 occlusion 프레임: {len(occluded_frames)}개")

    # 3단계: 보간 수행 및 저장
    print("  보간 및 저장 중...")
    occlusion_log = []

    for i in range(total):
        fname = os.path.basename(mask_files[i])

        if i in occluded_frames:
            occluded_region = occluded_frames[i]
            occluded_pixels = np.count_nonzero(occluded_region)
            orig_area = np.count_nonzero(all_masks[i])

            # 공간적 보간
            frame_bgr = cv2.imread(frame_files[i])
            spatial_gray = spatial_inpaint(frame_bgr, all_masks[i], occluded_region)

            # 시간적 보간
            temporal_mask = temporal_interpolate_mask(all_masks, list(range(total)), i)

            # 결합
            interp_mask = combine_interpolation(
                all_masks[i], spatial_gray, temporal_mask, occluded_region
            )

            interp_area = np.count_nonzero(interp_mask)
            restored_pixels = interp_area - orig_area

            occlusion_log.append({
                "frame": i,
                "fname": fname,
                "occluded_px": occluded_pixels,
                "restored_px": restored_pixels,
                "orig_area": orig_area,
                "interp_area": interp_area,
            })

            # 보간된 mask 저장
            cv2.imwrite(os.path.join(interp_dir, fname), interp_mask)

            # 비교 시각화 저장
            comparison = create_comparison(all_masks[i], interp_mask, i)
            cv2.imwrite(os.path.join(compare_dir, fname), comparison)
        else:
            # occlusion 없는 프레임은 원본 mask 그대로 복사
            cv2.imwrite(os.path.join(interp_dir, fname), all_masks[i])

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} 완료")

    # 결과 요약
    print(f"\n  [결과] occlusion 감지: {len(occlusion_log)}프레임")
    if occlusion_log:
        print(f"  {'프레임':<20} {'occluded(px)':<14} {'restored(px)':<14} {'원본면적':<12} {'보간면적':<12}")
        print(f"  {'-'*72}")
        for entry in occlusion_log[:20]:  # 최대 20개만 출력
            print(f"  {entry['fname']:<20} {entry['occluded_px']:<14} {entry['restored_px']:<14} "
                  f"{entry['orig_area']:<12} {entry['interp_area']:<12}")
        if len(occlusion_log) > 20:
            print(f"  ... 외 {len(occlusion_log) - 20}프레임 더")

    return occlusion_log


if __name__ == "__main__":
    print("Occlusion 보간 파라미터:")
    print(f"  area_drop_ratio={AREA_DROP_RATIO}, iou_threshold={IOU_THRESHOLD}")
    print(f"  min_occluded_px={MIN_OCCLUDED_PIXELS}, temporal_window={TEMPORAL_WINDOW}")
    print(f"  inpaint_radius={INPAINT_RADIUS}, dilation_kernel={DILATION_KERNEL}")

    all_logs = {}
    for target in TARGETS:
        log = process_target(target)
        all_logs[target["name"]] = log

    # 전체 요약
    print(f"\n{'='*60}")
    print("전체 요약")
    print(f"{'='*60}")
    for name, log in all_logs.items():
        frame_nums = [e["frame"] for e in log]
        print(f"\n[{name}] occlusion 프레임 {len(log)}개")
        if frame_nums:
            print(f"  프레임 번호: {frame_nums}")
