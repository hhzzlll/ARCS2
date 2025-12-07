import cv2
import os

# ================= 配置区域 =================
# 图片文件夹路径 (请修改此处为实际文件夹路径)
folder_path = r"D:\ARCS2\Project Code\Data\20251028\calibration\intrinsics"

# 旋转方向: 'clockwise' (顺时针90度) 或 'counterclockwise' (逆时针90度)
direction = 'clockwise' 
# ===========================================

# 支持的图片扩展名
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

rotate_code = cv2.ROTATE_90_CLOCKWISE if direction == 'clockwise' else cv2.ROTATE_90_COUNTERCLOCKWISE

if not os.path.exists(folder_path):
    print(f"错误: 找不到文件夹: {folder_path}")
    exit(1)

print(f"正在处理文件夹: {folder_path}")
print(f"方向: {direction}")

count = 0
for filename in os.listdir(folder_path):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in valid_extensions:
        continue

    file_path = os.path.join(folder_path, filename)
    
    # 读取图片
    img = cv2.imread(file_path)
    if img is None:
        continue

    # 旋转图片
    rotated_img = cv2.rotate(img, rotate_code)
    
    # 覆盖原图
    cv2.imwrite(file_path, rotated_img)
    print(f"已旋转并覆盖: {filename}")
    count += 1

print(f"完成。共处理 {count} 张图片。")
