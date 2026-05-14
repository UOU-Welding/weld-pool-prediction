import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def create_time_windows(features_array, window_size=5):
    """
    (N, Features) 데이터를 (N-W+1, W, Features) 형태의 Time-Window 텐서로 변환하는 함수
    """
    X = []
    
    # 시계열 슬라이딩 윈도우 생성
    for i in range(len(features_array) - window_size + 1):
        # i 부터 i+window_size-1 까지의 센서 데이터 묶음 (Input)
        window_X = features_array[i : i + window_size]

        X.append(window_X)
        
    return np.array(X)

def make_dataset(TF_case, txt_path):
    # 1. 데이터 로드
    heatmaps = np.load(os.path.join(base_dir, 'dataset', f'crop_heatmaps_{TF_case}.npy'))
    columns = ['용접전류', '용접전압', '송급속도', '용접저항']
    df = pd.read_csv(txt_path, header=1, names=columns, encoding='unicode_escape')
    sensors = df.values.copy()
    
    # 초기 1000 rows 제거 (0 ~ 0.4초)
    sensors = sensors[1000:]

    # 2. 주파수(FPS/Hz) 설정
    fps = 25.0   # 약 30.24 FPS
    sensor_hz = 2500.0
    pre_frame = 4

    # 3. 신호 추출 및 시간 축 생성
    max_temps = heatmaps.astype(np.float32).max(axis=(1, 2))
    current = sensors[:, 0]

    time_h = np.arange(len(max_temps)) / fps
    time_s = np.arange(len(current)) / sensor_hz

    # 4. 이벤트(용접 시작) 지점 탐색
    # 센서: 전류가 50A 이상으로 튀어오르는 첫 시점
    start_sensor_idx = np.where(current > 50.0)[0][0] - pre_frame*100
    start_sensor_time = time_s[start_sensor_idx + pre_frame*100]

    # 열화상: 최고 온도가 급격히 상승(예: 100도 이상)하는 첫 시점
    start_heatmap_idx = np.where(max_temps > 100.0)[0][0]
    start_heatmap_time = time_h[start_heatmap_idx]

    print(f"센서 수집 버튼 누른 후 용접 시작까지 걸린 시간: {start_sensor_time:.3f}초")
    print(f"카메라 녹화 버튼 누른 후 용접 시작까지 걸린 시간: {start_heatmap_time:.3f}초")
    print(f"발생한 시차(Offset): {abs(start_heatmap_time - start_sensor_time):.3f}초")

    # 5. GT 매칭을 위한 동기화 (예: 용접 시작부터 끝까지 매칭)
    # 센서에서 용접이 종료되는 시점(전류가 다시 50A 이하로 떨어짐)을 기준으로 삼음
    end_sensor_idx = np.where(current > 50.0)[0][-1]
    weld_duration = time_s[end_sensor_idx] - start_sensor_time # 실제 용접 수행 시간(초)

    print(f" 실제 용접 수행 시간(초): {weld_duration:.4f}")
    # 매칭할 프레임 길이 계산
    heatmap_weld_frames = int(weld_duration * fps)

    # 6. 최종적으로 매칭된 데이터 크롭(Crop)
    # 열화상: 시작 인덱스 ~ (시작 인덱스 + 용접 프레임수)
    aligned_heatmaps = heatmaps[start_heatmap_idx : start_heatmap_idx + heatmap_weld_frames]

    print(f"센서 시작 : {start_sensor_idx}, 센서 종료 : {end_sensor_idx}, \
            열화상 시작 :{start_heatmap_idx}, 열화상 종료 : {start_heatmap_idx + heatmap_weld_frames}")
    print(f"센서 구간 길이 {end_sensor_idx - start_sensor_idx}, \
           열화상 구간 길이 : {len(aligned_heatmaps)}")

    # 센서: 시작 시간 ~ 종료 시간 사이의 데이터를 FPS(25Hz)에 맞춰 리샘플링하여 매칭
    matched_sensor_list = []
    for i in range(heatmap_weld_frames+pre_frame):
        # 열화상 1프레임에 해당하는 시간
        target_time_in_weld = i / fps 
        
        # 센서 데이터상에서 해당 시간의 인덱스를 계산
        # (용접 시작 인덱스 + 타겟 시간에 해당하는 센서 인덱스 이동량)
        sensor_target_idx = start_sensor_idx + int(target_time_in_weld * sensor_hz)
        # print(sensor_target_idx)
        # 카메라 1프레임 동안 들어온 센서 데이터(100개) 묶음의 평균
        chunk_start = sensor_target_idx
        chunk_end = chunk_start + int(sensor_hz / fps)
        
        sensor_chunk_mean = sensors[chunk_start:chunk_end].mean(axis=0)
        matched_sensor_list.append(sensor_chunk_mean)

    aligned_sensors = np.array(matched_sensor_list)

    print("-" * 40)
    print(f"매칭 완료된 열화상 데이터 Shape: {aligned_heatmaps.shape}")
    print(f"매칭 완료된 센서 데이터 Shape  : {aligned_sensors.shape}")

    window_sensors = create_time_windows(aligned_sensors, window_size=5)
    
    print(f"윈도우 변환된 센서 데이터 Shape  : {window_sensors.shape}")

    return aligned_heatmaps, window_sensors


if __name__ == "__main__":
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = "../data/raw/T-Fillet/TF031_060"
    data_files = [
        f"TF{i:03d}" for i in range(31, 32)
    ]


    for case in data_files:
        Pass = f"{case}_1PASS"
        txt_path = f"{data_path}/{case}/{Pass}/{case}_wellteq.txt"
        aligned_heatmaps, window_sensors = make_dataset(case, txt_path)
        
        np.save(os.path.join(base_dir, 'dataset', f"aligned_heatmap_{case}.npy"), aligned_heatmaps)
        np.save(os.path.join(base_dir, 'dataset', f"window_sensors_{case}.npy"), window_sensors)