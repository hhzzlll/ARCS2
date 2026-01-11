import cv2
import numpy as np
import pandas as pd
import toml
from pathlib import Path

# 1. 配置路径
project_dir = Path(r"D:\ARCS2\Project Code\Data\20251230202711")
video_path = project_dir / "videos" / "cam01.mp4"
ground_csv_path = project_dir / "ground_data" / "ground.csv"
calib_path = project_dir / "calibration" / "Calib_scene.toml"
output_video_path = project_dir / "cam01_overlay.mp4"

# 手动对齐参数 (正值表示 Ground Truth 滞后，需要向左移；负值表示 Ground Truth 超前，需要向右移)
# 这里的 offset 是帧数
GROUND_OFFSET = 10#-40

# 2. 读取标定参数
calib_data = toml.load(calib_path)
# 假设 cam01 对应的是 'int_cam01_img'
# 注意：toml 结构可能不同，这里假设直接在根下或者在 'cameras' 下
# 根据提供的结构，通常是 calib_data['int_cam01_img'] 等

cam_key_int = 'int_cam01_img'

# 内参
K = np.array(calib_data[cam_key_int]['matrix'])
dist_coeffs = np.array(calib_data[cam_key_int]['distortions'])

# 外参 (世界 -> 相机)
# Pose2Sim 的外参通常是 R 和 T，将世界坐标转换到相机坐标
# P_cam = R * P_world + T
# 注意：在这个 toml 文件中，rotation 和 translation 直接在 [int_cam03_img] 下
R = np.array(calib_data[cam_key_int]['rotation'])
T = np.array(calib_data[cam_key_int]['translation'])

# 3. 读取 Ground Truth 数据
# ground.csv 前两行是 header，第三行是空行，第四行开始是数据
# 根据之前的逻辑，我们直接读取，跳过前几行
# 实际上 pandas read_csv 可以自动处理 header
# 观察 ground.csv 内容：
# Line 1: Frame#, Time, Hip, ... (列名)
# Line 2: ,,X1,Y1,Z1, ... (坐标轴)
# Line 3: (empty)
# Line 4: Data...
df = pd.read_csv(ground_csv_path, header=1) # 使用第二行作为 header (X1, Y1, Z1...)
# 注意：第一行是部位名称，第二行是 XYZ。我们需要结合起来或者只用 XYZ。
# 简单起见，我们假设列顺序是固定的，每 3 列一个点。
# 忽略前两列 (Frame#, Time) -> 实际上第二行前两列是空的
# 从第 3 列开始是坐标数据

# 4. 视频处理
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

frame_idx = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing video: {video_path}")
print(f"Total frames: {total_frames}")

# --- 数据预处理 ---
# 读取 CSV，跳过前 3 行 (header 2 lines + 1 empty line)
# 实际上 pandas read_csv header=1 会把第二行当做列名，第一行被忽略。
# 我们直接读取数据区域。
# 重新读取：
df_raw = pd.read_csv(ground_csv_path, header=None, skiprows=3)
# df_raw 的第 0 列是 Frame#
# 将 Frame# 设为索引
df_raw.set_index(0, inplace=True)

# --- 筛选特定部位 ---
# 读取 CSV Header 获取列索引
with open(ground_csv_path, 'r') as f:
    header_line = f.readline().strip().split(',')

# 用户需要的部位
target_parts = ['RShoulder', 'RElbow', 'RWrist', 'Neck', 'Hip', 'RAnkle', 'RHip', 'RKnee']
target_indices = []

print("Selected parts indices:")
for part in target_parts:
    try:
        # 找到部位名称在 header 中的索引
        idx = header_line.index(part)
        # 添加 X, Y, Z 的索引 (假设紧接着是 Y, Z)
        # df_raw set_index(0) 后，列索引与 header 索引对应
        target_indices.extend([idx, idx+1, idx+2])
        print(f"  {part}: {idx}, {idx+1}, {idx+2}")
    except ValueError:
        print(f"Warning: Part '{part}' not found in CSV header.")
# -----------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_frame_num = frame_idx # 假设视频从 0 开始，或者根据实际情况调整
    # ground.csv 的 Frame# 通常是从 0 或 1 开始的，且可能不是连续的或者有偏移
    # 这里假设视频帧号与 CSV 中的 Frame# 一致
    
    # 应用手动偏移
    ground_frame_idx = current_frame_num + GROUND_OFFSET
    
    if ground_frame_idx in df_raw.index:
        row = df_raw.loc[ground_frame_idx]
        
        # 提取指定列
        # 注意：row 的索引是整数列号
        try:
            coords = row[target_indices].values.astype(float)
        except KeyError as e:
            print(f"Error accessing columns: {e}")
            break
        
        # 重塑为 (N, 3)
        num_points = len(coords) // 3
        points_3d = coords.reshape((num_points, 3))

        # 坐标轴变换: Z->X, X->Y, Y->Z
        # 原来: 0->X, 1->Y, 2->Z
        # 现在: 0->Z, 1->X, 2->Y
        points_3d = points_3d[:, [2, 0, 1]]
        
        # 投影到 2D
        # P_cam = R * P_world + T
        # P_img = K * P_cam / Z
        
        # 批量投影
        # points_3d 是 (N, 3)
        # R 是 (3, 3), T 是 (3,)
        
        # 转换到相机坐标系
        # points_cam = (R @ points_3d.T).T + T
        
        # 投影到像素坐标系
        # points_img_homo = (K @ points_cam.T).T
        # points_img = points_img_homo[:, :2] / points_img_homo[:, 2:3]
        
        # 畸变校正 (可选，如果标定参数包含畸变且需要高精度)
        # cv2.projectPoints 可以处理畸变
        # 使用 cv2.projectPoints 更简单直接
        # 注意：cv2.Rodrigues 输入如果是旋转向量则输出旋转矩阵，反之亦然。
        # 这里 R 是旋转向量 (3,)，所以直接用。
        # 如果 R 是 (3,3) 矩阵，则需要转换。
        # Calib_scene.toml 中的 rotation 通常是旋转向量 (Rodrigues 形式)
        
        # 检查 R 的形状
        if R.shape == (3, 3):
            rvec, _ = cv2.Rodrigues(R)
        else:
            rvec = R
            
        img_points, _ = cv2.projectPoints(points_3d, rvec, T, K, dist_coeffs)
        img_points = img_points.reshape(-1, 2)
        
        # 绘制点
        for p in img_points:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) # 红色点
                
    out.write(frame)
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames", end='\r')

cap.release()
out.release()
print("\nDone.")
