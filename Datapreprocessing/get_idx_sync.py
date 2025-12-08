import toml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 读取 config.toml
config_path = Path('config/config.toml')
config = toml.load(config_path)

project_dir = Path(__file__).resolve().parent.parent
project_name = config['project']['project_name']

# 2. 解析路径
imu_path_template = config['estimation']['file_paths']['imu_farm_fpath']
imu_path = Path(imu_path_template.format(project_dir=project_dir, project_name=project_name))

img_path_template = config['estimation']['file_paths']['image_fpath']
img_path = Path(img_path_template.format(project_dir=project_dir, project_name=project_name))

# 3. 读取数据
print(f"Reading IMU data: {imu_path}")
df_imu = pd.read_csv(imu_path)
print(f"Reading Image data: {img_path}")
df_img = pd.read_csv(img_path)

# 存储点击结果
selected_indices = {'imu': None, 'img': None}

# --- IMU Plot ---
def on_click_imu(event):
    if event.xdata is not None:
        idx = int(event.xdata)
        selected_indices['imu'] = idx
        print(f"IMU Selected Index (idx_sync_w): {idx}")
        plt.close() # 关闭当前窗口，继续下一个

plt.figure(figsize=(12, 6))
plt.plot(df_imu.index, df_imu['Acc_X'], label='Acc_X', color='r')
plt.plot(df_imu.index, df_imu['Acc_Y'], label='Acc_Y', color='g')
plt.plot(df_imu.index, df_imu['Acc_Z'], label='Acc_Z', color='b')
plt.title("IMU Data (Click to select idx_sync_w)")
plt.legend()
plt.connect('button_press_event', on_click_imu)
plt.show()

# --- Image Plot ---
def on_click_img(event):
    if event.xdata is not None:
        idx = int(event.xdata)
        selected_indices['img'] = idx
        print(f"Image Selected Index (idx_sync_kpts): {idx}")
        plt.close()

if selected_indices['imu'] is not None:
    plt.figure(figsize=(12, 6))
    # 假设 CSV 中列名为 'rwrist u' 和 'rwrist v'，如果不同请调整
    plt.plot(df_img['index'], df_img['rwrist u'], label='rwrist u', color='orange')
    plt.plot(df_img['index'], df_img['rwrist v'], label='rwrist v', color='purple')
    plt.title("Image Keypoints (Click to select idx_sync_kpts)")
    plt.legend()
    plt.connect('button_press_event', on_click_img)
    plt.show()

# 4. 更新 config.toml
if selected_indices['imu'] is not None and selected_indices['img'] is not None:
    config['estimation']['other_parameters']['idx_sync_w'] = selected_indices['imu']
    config['estimation']['other_parameters']['idx_sync_kpts'] = selected_indices['img']
    
    with open(config_path, 'w', encoding='utf-8') as f:
        toml.dump(config, f)
    
    print("\nConfig updated successfully!")
    print(f"idx_sync_w = {selected_indices['imu']}")
    print(f"idx_sync_kpts = {selected_indices['img']}")
else:
    print("\nSelection incomplete. Config not updated.")