import pandas as pd

# 文件路径 (请根据需要修改)
file_path = r"D:\ARCS2\Project Code\Data\20260117162440\imu_data\farm.csv"

# 设置读取行数: None 表示读取所有行，整数 n 表示读取前 n 行
# 例如: n_rows = 100 (前100行), n_rows = None (所有行)
n_rows = 150 

# 读取CSV
if n_rows:
    df = pd.read_csv(file_path, nrows=n_rows)
    print(f"正在计算前 {n_rows} 行的数据...")
else:
    df = pd.read_csv(file_path)
    print("正在计算所有行的数据...")

# 计算平均值
avg_x = df['Gyr_X'].mean()
avg_y = df['Gyr_Y'].mean()
avg_z = df['Gyr_Z'].mean()

print(f"文件: {file_path}")
print(f"Gyr_X 平均值: {avg_x}")
print(f"Gyr_Y 平均值: {avg_y}")
print(f"Gyr_Z 平均值: {avg_z}")
print(f"[{avg_x}, {avg_y}, {avg_z}]")