import pickle
import numpy as np
import os


def make_crop_heatmaps(data: dict, crop_height: int = 50, crop_width: int = 50, min_temp: float = 800.0) -> tuple[np.ndarray, list]:
    """
    최고 온도가 min_temp 이상인 프레임만 최고 온도 픽셀 중심으로 crop_height x crop_width 크롭.

    Parameters
    ----------
    data      : {key: 2D ndarray} 형태의 프레임 딕셔너리
    crop_height : 크롭 높이, 기본값 50
    crop_width  : 크롭 너비, 기본값 50
    min_temp  : 최고 온도 필터 임계값, 기본값 800.0

    Returns
    -------
    crops   : ndarray, shape = (N, crop_height, crop_width)
    indices : list of (frame_index, key) — 저장된 프레임의 순번과 원본 키
    """

    # 5. 30x30 픽셀 영역 자르기 위한 경계 좌표 계산 (반지름 15픽셀)
    crop_height = 30
    crop_width = 30

    crops = []
    indices = []  # (frame_index, key) 튜플 리스트

    for frame_index, (key, frame) in enumerate(data.items()):
        # 최고 온도가 임계값 미만이면 스킵
        if np.max(frame) < min_temp:
            continue

        h, w = frame.shape

        # 최고 온도 픽셀 좌표
        max_y, max_x = np.unravel_index(np.argmax(frame), frame.shape)

        # 크롭 경계 계산
        y1 = max_y - crop_height // 2 - 5
        y2 = max_y + crop_height // 2 - 5
        x1 = max_x - crop_width // 2
        x2 = max_x + crop_width // 2

        # 가장자리 보정
        if y1 < 0:
            y2 -= y1;  y1 = 0
        if y2 > h:
            y1 -= (y2 - h);  y2 = h
        if x1 < 0:
            x2 -= x1;  x1 = 0
        if x2 > w:
            x1 -= (x2 - w);  x2 = w

        crops.append(frame[y1:y2, x1:x2])
        indices.append((frame_index, key))

    if not crops:
        raise ValueError(f"최고 온도가 {min_temp} 이상인 프레임이 없습니다.")

    return np.stack(crops, axis=0), np.array(indices)


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Stage 1: CAE pre-training
    data_files = [
        f"TF{i:03d}" for i in range(31, 61)
    ]

    for tf_id in data_files:
        pkl_path = os.path.normpath(os.path.join(
            base_dir, '..', '..', 'data', 'raw', 'T-Fillet', 'TF031_060',
            tf_id,               # ex) "TF033"
            tf_id + '_1PASS',    # ex) "TF033_1PASS"
            tf_id + '_FLIR.pkl'  # ex) "TF033_FLIR.pkl"
        ))
        print(pkl_path)  # 확인용


        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 사용 예시
        crop_heatmaps, indices = make_crop_heatmaps(data, crop_height=14, crop_width=22, min_temp=800.0)

        print(f"=== {tf_id} ===")
        print(f"저장된 프레임 수: {len(indices)} / 전체 {len(data)}")
        print(f"ndarray shape  : {crop_heatmaps.shape}")
        print()
        # print("[ 저장된 프레임 인덱스 정보 ]")
        # for crop_idx, (frame_index, key) in enumerate(indices):
        #     print(f"  crops[{crop_idx:>4}] ← data 순번 {frame_index:>5} | key: {key}")

        np.save(os.path.join(base_dir, 'dataset', f"crop_heatmaps_{tf_id}.npy"), crop_heatmaps)