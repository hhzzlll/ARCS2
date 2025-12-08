import cv2
import os

# ================= 配置区域 =================
# 输入视频文件的绝对路径
input_path = r"D:\ARCS2\Project Code\Data\20251204173454\cam3.mp4"

# 输出视频文件的绝对路径
output_path = r"D:\ARCS2\Project Code\Data\20251204173454\cam03.mp4"

# 旋转方向: 'clockwise' (顺时针90度) 或 'counterclockwise' (逆时针90度)
direction = 'clockwise' 
# ===========================================

if not os.path.exists(input_path):
    print(f"错误: 找不到输入文件: {input_path}")
    exit(1)

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print(f"错误: 无法打开视频文件: {input_path}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 使用 mp4v 编码
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 交换宽高，因为旋转了90度
out = cv2.VideoWriter(output_path, fourcc, fps, (height, width))

rotate_code = cv2.ROTATE_90_CLOCKWISE if direction == 'clockwise' else cv2.ROTATE_90_COUNTERCLOCKWISE

print(f"正在处理: {input_path}")
print(f"输出到: {output_path}")
print(f"方向: {direction}")
print("开始转换...")

count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rotated_frame = cv2.rotate(frame, rotate_code)
    out.write(rotated_frame)
    
    count += 1
    if count % 100 == 0:
        print(f"已处理帧数: {count}/{total_frames}", end='\r')

cap.release()
out.release()
print(f"\n完成! 视频已保存至: {output_path}")
