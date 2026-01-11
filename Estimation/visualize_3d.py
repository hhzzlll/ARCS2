import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
import sys
import os

# 添加当前目录到 sys.path 以便导入 main1
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 临时屏蔽 plt.show 以免 main1.py 阻塞
original_show = plt.show
plt.show = lambda: None

try:
    import main1
finally:
    # 恢复 plt.show
    plt.show = original_show

def visualize_comparison(data):
    """
    可视化 Est, Int, IMU, Ground 四种结果的对比，并提供进度条
    """
    # 提取数据 (3, N)
    # 1. Est
    uarm_est = data.uarm_est
    farm_est = data.farm_est
    
    # 2. Int
    uarm_int = data.uarm_int
    farm_int = data.farm_int
    
    # 3. IMU
    uarm_imu = data.uarm_imu
    farm_imu = data.farm_imu
    
    # 4. Ground
    uarm_ground = data.uarm_ground
    farm_ground = data.farm_ground
    
    # 假设臂长
    L_uarm = 0.3
    L_farm = 0.25
    
    num_frames = uarm_est.shape[1]
    
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25) # 为滑块留出空间
    
    # 设置坐标轴范围
    limit = 0.6
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Arm Pose Comparison: Est vs Int vs IMU vs Ground')
    
    # 初始化线条
    ground_offset = 0#-45
    # 格式: (uarm_vec, farm_vec, color, label, style, offset)
    methods = [
        (uarm_est, farm_est, 'r', 'Est', '-', 0),
        (uarm_int, farm_int, 'b', 'Int', '--', 0),
        (uarm_imu, farm_imu, 'g', 'IMU', '-.', 0),
        (uarm_ground, farm_ground, 'k', 'Ground', ':', ground_offset)
    ]
    
    lines = []
    for _, _, color, label, style, _ in methods:
        line, = ax.plot([], [], [], color=color, label=label, linestyle=style, marker='o', lw=2)
        lines.append(line)
        
    ax.legend()
    
    # 文本显示当前帧
    text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def get_coords(uarm_vec, farm_vec, frame_idx):
        # 检查索引边界
        if frame_idx < 0 or frame_idx >= uarm_vec.shape[1]:
            return None, None, None
            
        # 检查 NaN (针对 Ground Truth)
        u = uarm_vec[:, frame_idx]
        f = farm_vec[:, frame_idx]
        if np.isnan(u).any() or np.isnan(f).any():
            return None, None, None
            
        shoulder = np.array([0, 0, 0])
        elbow = shoulder + u * L_uarm
        wrist = elbow + f * L_farm
        
        xs = [shoulder[0], elbow[0], wrist[0]]
        ys = [shoulder[1], elbow[1], wrist[1]]
        zs = [shoulder[2], elbow[2], wrist[2]]
        return xs, ys, zs

    def update(val):
        frame = int(slider.val)
        text.set_text(f"Frame: {frame}/{num_frames}")
        
        for i, (u_vec, f_vec, _, _, _, offset) in enumerate(methods):
            xs, ys, zs = get_coords(u_vec, f_vec, frame + offset)
            if xs is not None:
                lines[i].set_data(xs, ys)
                lines[i].set_3d_properties(zs)
            else:
                # 如果数据无效（如NaN），隐藏线条
                lines[i].set_data([], [])
                lines[i].set_3d_properties([])
                
        fig.canvas.draw_idle()

    # 添加滑块
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
    slider.on_changed(update)
    
    # 初始化第一帧
    update(0)
    
    plt.show()

if __name__ == "__main__":
    print("正在运行 main1.py 获取数据...")
    # 获取数据
    data = main1.main()
    
    if data is None:
        print("错误：未能获取数据。")
    else:
        visualize_comparison(data)
