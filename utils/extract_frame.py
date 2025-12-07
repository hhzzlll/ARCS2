import cv2
import os

def extract_first_frame(video_path, output_path):
    """
    从视频中提取第一帧并保存为图片
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 找不到视频文件 -> {video_path}")
        return

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return

    # 读取第一帧
    ret, frame = cap.read()

    if ret:
        # 保存帧为图片
        cv2.imwrite(output_path, frame)
        print(f"成功: 第一帧已保存到 -> {output_path}")
    else:
        print("错误: 无法读取第一帧")

    # 释放资源
    cap.release()

if __name__ == "__main__":
    # --- 配置路径 ---
    # 请修改为你实际的视频路径
    # 例如: r"Data\20251028\videos\cam1.mp4"
    video_file = r"D:\ARCS2\Project Code\Data\20251205\videos\extrinsic\cam3-2025-12-05 16-47-38.mp4"
    
    # 输出图片的文件名
    output_image = r"D:\ARCS2\Project Code\Data\20251205170451\calibration\extrinsics\ext_cam03_img\cam03.png"
    
    extract_first_frame(video_file, output_image)