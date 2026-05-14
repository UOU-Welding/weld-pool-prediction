"""
FLIR 열화상 프레임 고온 영역 segmentation 스크립트
- Otsu thresholding + morphological operation으로 고온 영역 추출
- binary mask 및 overlay 시각화 생성
"""

import cv2
import numpy as np
import os
import sys
import glob

# ============================================================
# 파라미터 설정 (필요시 조정)
# ============================================================
THRESHOLD_METHOD = "otsu"        # "otsu" 또는 "adaptive"
MANUAL_THRESHOLD = None          # None이면 자동, 숫자 지정 시 수동 threshold
MORPH_KERNEL_SIZE = 5            # morphological operation 커널 크기
MORPH_OPEN_ITER = 2              # opening 반복 (노이즈 제거)
MORPH_CLOSE_ITER = 2             # closing 반복 (구멍 메우기)
MIN_CONTOUR_AREA = 100           # 최소 contour 면적 (이하 제거)
OVERLAY_ALPHA = 0.4              # overlay 투명도 (0~1)
OVERLAY_COLOR = (0, 0, 255)      # overlay 색상 (BGR: 빨간색)
# ============================================================

BASE_RESULT = "/data/KHD/workspace/2026용접/결과"

TARGETS = [
    {
        "name": "TF016_FLIR",
        "frames_dir": f"{BASE_RESULT}/260325/TF016_FLIR.mp4_frames",
        "masks_dir":  f"{BASE_RESULT}/260325/TF016_FLIR.mp4_masks",
        "overlay_dir": f"{BASE_RESULT}/260325/TF016_FLIR.mp4_overlay",
    },
    {
        "name": "TF027_FLIR",
        "frames_dir": f"{BASE_RESULT}/260327/TF027_FLIR.mp4_frames",
        "masks_dir":  f"{BASE_RESULT}/260327/TF027_FLIR.mp4_masks",
        "overlay_dir": f"{BASE_RESULT}/260327/TF027_FLIR.mp4_overlay",
    },
    {
        "name": "TF030_FLIR",
        "frames_dir": f"{BASE_RESULT}/260327/TF030_FLIR.mp4_frames",
        "masks_dir":  f"{BASE_RESULT}/260327/TF030_FLIR.mp4_masks",
        "overlay_dir": f"{BASE_RESULT}/260327/TF030_FLIR.mp4_overlay",
    },
]


def segment_frame(image, method=THRESHOLD_METHOD, manual_thresh=MANUAL_THRESHOLD):
    """열화상 프레임에서 고온 영역을 segmentation"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) Thresholding
    if manual_thresh is not None:
        _, mask = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY)
        thresh_val = manual_thresh
    elif method == "otsu":
        thresh_val, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=51, C=-10
        )
        thresh_val = -1  # adaptive는 단일 값 없음
    else:
        raise ValueError(f"Unknown method: {method}")

    # 2) Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)

    # 3) 작은 contour 제거
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    return clean_mask, thresh_val


def create_overlay(image, mask, alpha=OVERLAY_ALPHA, color=OVERLAY_COLOR):
    """원본 이미지 위에 mask를 반투명 색상으로 overlay"""
    overlay = image.copy()
    overlay[mask > 0] = color
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result


def process_target(target):
    """하나의 영상 데이터셋 처리"""
    name = target["name"]
    frames_dir = target["frames_dir"]
    masks_dir = target["masks_dir"]
    overlay_dir = target["overlay_dir"]

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    total = len(frame_files)

    if total == 0:
        print(f"[SKIP] {name}: 프레임 없음 ({frames_dir})")
        return []

    print(f"\n{'='*60}")
    print(f"[{name}] {total}프레임 segmentation 시작")
    print(f"  프레임: {frames_dir}")
    print(f"  마스크: {masks_dir}")
    print(f"  오버레이: {overlay_dir}")
    print(f"{'='*60}")

    samples = []

    for i, fpath in enumerate(frame_files):
        fname = os.path.basename(fpath)
        image = cv2.imread(fpath)

        mask, thresh_val = segment_frame(image)

        # 마스크 저장
        cv2.imwrite(os.path.join(masks_dir, fname), mask)

        # 오버레이 저장
        overlay = create_overlay(image, mask)
        cv2.imwrite(os.path.join(overlay_dir, fname), overlay)

        # 샘플 수집 (균등 간격으로 5장)
        sample_indices = [int(total * j / 5) for j in range(5)]
        if i in sample_indices:
            hot_pixels = np.count_nonzero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            hot_ratio = hot_pixels / total_pixels * 100
            samples.append({
                "frame": fname,
                "thresh": thresh_val,
                "hot_ratio": hot_ratio,
                "mask_path": os.path.join(masks_dir, fname),
                "overlay_path": os.path.join(overlay_dir, fname),
            })

        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} 완료")

    print(f"[{name}] 완료!")
    return samples


if __name__ == "__main__":
    print(f"파라미터: method={THRESHOLD_METHOD}, manual_thresh={MANUAL_THRESHOLD}")
    print(f"  morph_kernel={MORPH_KERNEL_SIZE}, open_iter={MORPH_OPEN_ITER}, close_iter={MORPH_CLOSE_ITER}")
    print(f"  min_contour_area={MIN_CONTOUR_AREA}, overlay_alpha={OVERLAY_ALPHA}")

    all_samples = {}
    for target in TARGETS:
        samples = process_target(target)
        all_samples[target["name"]] = samples

    # 샘플 결과 출력
    print(f"\n{'='*60}")
    print("샘플 결과 요약")
    print(f"{'='*60}")
    for name, samples in all_samples.items():
        print(f"\n[{name}]")
        for s in samples:
            print(f"  {s['frame']}: thresh={s['thresh']:.0f}, 고온영역={s['hot_ratio']:.2f}%")
