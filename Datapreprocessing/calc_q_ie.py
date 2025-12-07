import toml
import pandas as pd
import numpy as np
from pathlib import Path

config_path = Path('config/config.toml')
config = toml.load(config_path)

project_dir = Path(__file__).resolve().parent.parent
project_name = config['project']['project_name']

path_template = config['estimation']['file_paths']['imu_board_fpath']
imu_board_path = Path(path_template.format(project_dir=project_dir, project_name=project_name))

print(f"Reading IMU data from: {imu_board_path}")

df = pd.read_csv(imu_board_path)

q_w = df['Quat_W'].mean()
q_x = df['Quat_X'].mean()
q_y = df['Quat_Y'].mean()
q_z = df['Quat_Z'].mean()

norm = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)

q_ie = [float(q_w/norm), float(q_x/norm), float(q_y/norm), float(q_z/norm)]

print(f"Calculated average q_ie: {q_ie}")

config['estimation']['transform_quaternion']['q_ie'] = q_ie
with open(config_path, 'w') as f:
    toml.dump(config, f)
