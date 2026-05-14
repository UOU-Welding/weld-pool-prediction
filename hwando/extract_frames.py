"""
TF016_FLIR.mp4 열화상 영상 프레임 추출 스크립트
- 입력: /data/KHD/workspace/2026용접/데이터/260325/TF016_FLIR.mp4
- 출력: /data/KHD/workspace/2026용접/결과/260325/TF016_FLIR.mp4_frames/frame_XXXX.png
"""

import cv2
import os
import sys

VIDEO_PATH = "/data/KHD/workspace/2026용접/데이터/260325/TF016_FLIR.mp4"
OUTPUT_DIR = "/data/KHD/workspace/2026용접/결과/260325/TF016_FLIR.mp4_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] 영상을 열 수 없습니다: {VIDEO_PATH}")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"영상 정보: {width}x{height}, {fps:.2f}fps, 총 {total}프레임")
print(f"출력 경로: {OUTPUT_DIR}")
print("프레임 추출 중...")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    filename = os.path.join(OUTPUT_DIR, f"frame_{count:04d}.png")
    cv2.imwrite(filename, frame)
    count += 1
    if count % 100 == 0:
        print(f"  {count}/{total} 완료")

cap.release()
print(f"\n추출 완료: 총 {count}프레임 저장됨")
print(f"출력 디렉토리: {OUTPUT_DIR}")

# 샘플 프레임 표시
sample_path = os.path.join(OUTPUT_DIR, "frame_0000.png")
if os.path.exists(sample_path):
    sample = cv2.imread(sample_path)
    h, w = sample.shape[:2]
    size_kb = os.path.getsize(sample_path) / 1024
    print(f"\n샘플 프레임 (frame_0000.png): {w}x{h}, {size_kb:.1f}KB")
