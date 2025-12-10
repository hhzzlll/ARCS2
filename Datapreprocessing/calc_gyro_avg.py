import pandas as pd

# 文件路径 (请根据需要修改)
file_path = r"D:\论文_课设_报告等\实习\Movella_DOT_Data_Exporter-2023.6.0-Windows\data\20251209_200658\farm_D422CD00810E_20251209_200554.csv"

# 读取CSV
df = pd.read_csv(file_path)

# 计算平均值
avg_x = df['Gyr_X'].mean()
avg_y = df['Gyr_Y'].mean()
avg_z = df['Gyr_Z'].mean()

print(f"文件: {file_path}")
print(f"Gyr_X 平均值: {avg_x}")
print(f"Gyr_Y 平均值: {avg_y}")
print(f"Gyr_Z 平均值: {avg_z}")
print(f"[{avg_x}, {avg_y}, {avg_z}]")