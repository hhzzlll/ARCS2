import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
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

def visualize_arm_3d(data):
    """
    可视化估计的大小臂三维姿态
    """
    uarm_est = data.uarm_est  # (3, N)
    farm_est = data.farm_est  # (3, N)
    
    # 假设臂长 (单位: m，或者相对单位)
    L_uarm = 0.3
    L_farm = 0.25
    
    # 降采样以提高动画流畅度
    step = 5
    uarm_est = uarm_est[:, ::step]
    farm_est = farm_est[:, ::step]
    num_frames = uarm_est.shape[1]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    limit = 0.6
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Estimated Arm Pose 3D Visualization')
    
    # 初始化线条和点
    line, = ax.plot([], [], [], 'o-', lw=2)
    
    # 文本显示当前帧
    text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        text.set_text("")
        return line, text

    def update(frame):
        # 肩关节位置 (原点)
        shoulder = np.array([0, 0, 0])
        
        # 肘关节位置
        # 注意：uarm_est 是方向向量，需要乘以长度
        # 另外需要确认 uarm_est 的方向定义。通常是从肩指向肘，或者反之。
        # main1.py 中: uarm_est[:, i] = (T_es @ lb).flatten(), lb=[1,0,0]
        # 假设 lb=[1,0,0] 指向骨骼长轴方向。
        elbow = shoulder + uarm_est[:, frame] * L_uarm
        
        # 腕关节位置
        wrist = elbow + farm_est[:, frame] * L_farm
        
        # 组合坐标
        xs = [shoulder[0], elbow[0], wrist[0]]
        ys = [shoulder[1], elbow[1], wrist[1]]
        zs = [shoulder[2], elbow[2], wrist[2]]
        
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        
        text.set_text(f"Frame: {frame * step}")
        return line, text

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=30)
    
    plt.show()

if __name__ == "__main__":
    print("正在运行 main1.py 获取数据...")
    # 获取数据
    data = main1.main()
    
    if data is None:
        print("错误：未能获取数据。")
    else:
        print("数据获取成功，开始可视化...")
        visualize_arm_3d(data)
