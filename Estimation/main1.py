import json
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import sys
from numpy.typing import NDArray

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, 'utilspy')
sys.path.append(utils_dir)

from utilspy.angle_between_vectors import angle_between_vectors
from utilspy.qInv import qInv
from utilspy.qMul import qMul
from utilspy.transFromQuat import transFromQuat
from estimatePose import estimatePose
from myStateTransitionFcn import myStateTransitionFcn
from myMeasurementLikelihoodFcn import myMeasurementLikelihoodFcn
from types_def import (
    Config,
    Data,
)

def _load_toml(path: Path) -> Dict[str, Any]:
    import toml 
    return toml.load(str(path))


def _guess_app_config_path() -> Path | None:
    base = Path(__file__).resolve().parent.parent
    candidates = [
        base / 'config' / 'config.toml',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config(config_path: str | None = None) -> Config:
    app_cfg_path = _guess_app_config_path()
    if app_cfg_path and app_cfg_path.suffix.lower() in ('.toml',):
        data = _load_toml(app_cfg_path)
        est = data.get('estimation') if isinstance(data, dict) else None
        prj = data.get('project') if isinstance(data, dict) else None
        if isinstance(est, dict):
            d = est  
            project_dir: str | None = None
            if isinstance(prj, dict):
                raw_dir = prj.get('project_dir')
                raw_name = prj.get('project_name')
                # 解析 project_dir
                if isinstance(raw_dir, str) and raw_dir.strip():
                    p = Path(raw_dir.strip())
                    if not p.is_absolute():
                        p = (Path(__file__).resolve().parent.parent / p).resolve()
                    project_dir = str(p)
                elif isinstance(raw_name, str) and raw_name.strip():
                    p = Path(__file__).resolve().parent.parent
                    project_dir = str(p)

            if isinstance(d.get('file_paths'), dict):
                fp = d['file_paths']
                p_name = str(prj.get('project_name', '')) if isinstance(prj, dict) else ''
                for k, v in list(fp.items()):
                    if isinstance(v, str):
                        if project_dir and '{project_dir}' in v:
                            v = v.replace('{project_dir}', project_dir)
                        if p_name and '{project_name}' in v:
                            v = v.replace('{project_name}', p_name)
                        fp[k] = v

            cfg: Config = Config.from_dict(d)
            cfg.validate()
            return cfg

    raise FileNotFoundError('Configuration not found: Please provide estimation.* configuration in config/config.toml')

def state_transition_adapter(state: NDArray[np.float64],
                             angular_velocity: NDArray[np.float64],
                             dt: float,
                             _unused: object) -> NDArray[np.float64]:
    
    w = angular_velocity
    w_norm = np.linalg.norm(w)
    
    if w_norm < 1e-6:
        return state  
    
    w_unit = w / w_norm
    angle = w_norm * dt
    
    dq = np.array([
        np.cos(angle/2),
        w_unit[0] * np.sin(angle/2),
        w_unit[1] * np.sin(angle/2), 
        w_unit[2] * np.sin(angle/2)
    ])
    
    from utilspy.qMul import qMul
    new_state = qMul(state.reshape(-1, 1), dq.reshape(-1, 1)).flatten()
    
    new_state = new_state / np.linalg.norm(new_state)
    
    return new_state

def main():
    config = load_config()
    
    file_paths = config.file_paths
    camera_params = config.camera_parameters
    transformation_matrix = config.transformation_matrix
    transform_quaternion = config.transform_quaternion
    other_params = config.other_parameters
    
    imu_farm_fpath = file_paths.imu_farm_fpath
    imu_uarm_fpath = file_paths.imu_uarm_fpath
    image_fpath = file_paths.image_fpath
    ground_fpath = file_paths.ground_fpath
    
    fx = camera_params.fx
    fy = camera_params.fy
    
    T_cw = np.array(transformation_matrix.T_cw)
    
    q_ie = np.array(transform_quaternion.q_ie).reshape(-1, 1)
    
    numParticles = other_params.num_particles
    hz_imu = other_params.hz_imu
    hz_image = other_params.hz_image
    idx_sync_w = other_params.idx_sync_w
    idx_sync_kpts = other_params.idx_sync_kpts
    
    # Manual alignment parameter 
    # positive value means Ground Truth lags and needs to be shifted left; 
    # negative value means Ground Truth leads and needs to be shifted right)
    ground_offset = -8 # 23
    
    np.random.seed(0)
    
    print("Configuration parameters loaded:")
    print(f"IMU Frequency: {hz_imu} Hz")
    print(f"Image Frequency: {hz_image} Hz")
    print(f"Number of Particles: {numParticles}")
    print(f"Sync Indices - IMU: {idx_sync_w}, Keypoints: {idx_sync_kpts}")
    
    print("Loading IMU and image data...")
    
    imu_farm_df = pd.read_csv(imu_farm_fpath)
    imu_uarm_df = pd.read_csv(imu_uarm_fpath)
    image_df = pd.read_csv(image_fpath)
    
    imu_farm_df = imu_farm_df.select_dtypes(include=[np.number])
    imu_uarm_df = imu_uarm_df.select_dtypes(include=[np.number])

    gyr_cols = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']

    gyr_farm = imu_farm_df[gyr_cols].to_numpy(dtype=float)
    gyr_uarm = imu_uarm_df[gyr_cols].to_numpy(dtype=float)
    image_df = image_df.select_dtypes(include=[np.number])
    
    imu_farm = imu_farm_df.values.astype(float)
    imu_uarm = imu_uarm_df.values.astype(float)
    image_data = image_df.values.astype(float)
    
    kpts_shoulder_elbow = image_data[:, 1:7]  # [u,v,conf] for shoulder-elbow
    kpts_elbow_wrist = image_data[:, 4:10]    # [u,v,conf] for elbow-wrist
    
    t_imu_farm = imu_farm[:, 1].astype(float)
    quat_farm = imu_farm[:, 2:6].T.astype(float)  
    
    t_imu_uarm = imu_uarm[:, 1].astype(float)
    quat_uarm = imu_uarm[:, 2:6].T.astype(float) 
    
    print("Loading ground truth data...")
    motion_df = pd.read_csv(ground_fpath)
    print(f"Ground truth data shape: {motion_df.shape}")
        
    motion = motion_df.iloc[1:, :].values 
        
    
    motion_numeric = []
    for i in range(motion.shape[0]):
        row = []
        for j in range(motion.shape[1]):
            val = float(motion[i, j]) if motion[i, j] != '' else 0.0
            row.append(val)
        motion_numeric.append(row)
    motion = np.array(motion_numeric, dtype=float)
        
    print(f"Processed ground truth data shape: {motion.shape}")

    
    offset = 2
    keyPointList = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe', 'RSmallToe', 'RHeel', 
                    'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 
                    'Neck', 'Head', 'Nose', 
                    'RShoulder', 'RElbow', 'RWrist', 
                    'LShoulder', 'LElbow', 'LWrist']
    
    # each keypoint has 3 values (X,Y,Z)
    idxList = list(range(offset, offset + len(keyPointList) * 3, 3))
    
    keyPointIndex_XYZ = dict(zip(keyPointList, idxList))
    
    sz_ground = len(motion)
    uarm_ground = np.full((3, sz_ground), np.nan)
    farm_ground = np.full((3, sz_ground), np.nan)
    angle_ground = np.full(sz_ground, np.nan)
    print("Calculating ground truth data...")
    for i in range(sz_ground):
        idx = keyPointIndex_XYZ['RShoulder']
        rshoulder = motion[i, idx:idx+3].astype(float)
        
        idx = keyPointIndex_XYZ['RElbow']
        relbow = motion[i, idx:idx+3].astype(float)
        
        idx = keyPointIndex_XYZ['RWrist']
        rwrist = motion[i, idx:idx+3].astype(float)
        
        uarm = (relbow - rshoulder)[[2, 0, 1]]  # Z->X, X->Y, Y->Z
        farm = (rwrist - relbow)[[2, 0, 1]]

        uarm_norm = np.linalg.norm(uarm)
        farm_norm = np.linalg.norm(farm)
        
        uarm_ground[:, i] = uarm / uarm_norm
        farm_ground[:, i] = farm / farm_norm
        angle_ground[i] = angle_between_vectors(-uarm, farm)
    
    print("Syncing data...")
    t_sync_imu = t_imu_farm[idx_sync_w - 1] 
    f_imu = 1 / hz_imu
    dt_imu = f_imu / (1 / 1000000)  # [1 tick = 1s/1M = 1us]
    
    f_image = 1 / hz_image
    dt_image = f_image / (1 / 1000000)  # [1 tick = 1s/1M = 1us]
    t0 = t_sync_imu - dt_image * (idx_sync_kpts - 1)
    n_image = len(image_data)
    # t_image = np.arange(t0, t0 + n_image * dt_image, dt_image)
    temp_array = np.arange(t0, t0 + n_image * dt_image, dt_image)
    t_image = temp_array[:n_image]
    
    idx_imu_start = 0  
    q_se_farm = quat_farm[:, idx_imu_start:idx_imu_start+1]
    q_se_uarm = quat_uarm[:, idx_imu_start:idx_imu_start+1]
    q_es_farm = qInv(q_se_farm)
    q_es_uarm = qInv(q_se_uarm)
    
    q_ei = qInv(q_ie)
    q0_farm = qMul(q_ei, q_se_farm)
    q0_uarm = qMul(q_ei, q_se_uarm)
    
    bias_farm = np.deg2rad(np.array([0.3022778628443215, -1.5259842610133334, -0.26910324191867685]))
    bias_uarm = np.deg2rad(np.array([1.2221464238998792, -1.2663199782371521, 1.4864542516072592]))
    
    w_se_farm = np.column_stack([
        t_imu_farm[idx_imu_start:] * 1e-6,
        np.deg2rad(gyr_farm[idx_imu_start:, :])  
    ])
    w_se_farm[:, 1] -= bias_farm[0]
    w_se_farm[:, 2] -= bias_farm[1]
    w_se_farm[:, 3] -= bias_farm[2]
    
    w_se_uarm = np.column_stack([
        t_imu_farm[idx_imu_start:] * 1e-6,
        np.deg2rad(gyr_uarm[idx_imu_start:, :]) 
    ])
    w_se_uarm[:, 1] -= bias_uarm[0]
    w_se_uarm[:, 2] -= bias_uarm[1]
    w_se_uarm[:, 3] -= bias_uarm[2]
    
    kpts_uarm = np.column_stack([t_image * 1e-6, kpts_shoulder_elbow])
    kpts_farm = np.column_stack([t_image * 1e-6, kpts_elbow_wrist])    
    
    q0_uarm_flat = q0_uarm.flatten()
    q0_farm_flat = q0_farm.flatten()
    # q0_uarm_flat = np.array([ 0.68962683,  0.,  -0.71609053, 0.10783867]).reshape(4, 1).flatten()
    # q0_farm_flat = np.array([ 0.60959801, 0., -0.78576101, -0.10473727]).reshape(4, 1).flatten()

    # pure integration without filtering
    noFilter = True
    int_uarm = estimatePose(q0_uarm_flat, numParticles, w_se_uarm, kpts_uarm, fx, fy, T_cw, 
                           myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter)
    int_farm = estimatePose(q0_farm_flat, numParticles, w_se_farm, kpts_farm, fx, fy, T_cw, 
                           myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter)

    # use particle filter
    noFilter = False
    qEst_uarm = estimatePose(q0_uarm_flat, numParticles, w_se_uarm, kpts_uarm, fx, fy, T_cw, 
                            myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter)
    qEst_farm = estimatePose(q0_farm_flat, numParticles, w_se_farm, kpts_farm, fx, fy, T_cw, 
                            myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter)

    sz_imu = qEst_uarm.shape[1]
    
    farm_int = np.full((3, sz_imu), np.nan)
    uarm_int = np.full((3, sz_imu), np.nan)
    farm_est = np.full((3, sz_imu), np.nan)
    uarm_est = np.full((3, sz_imu), np.nan)
    farm_imu = np.full((3, sz_imu), np.nan)
    uarm_imu = np.full((3, sz_imu), np.nan)

    R_uarm_int = np.full((sz_imu, 3, 3), np.nan)
    R_farm_int = np.full((sz_imu, 3, 3), np.nan)
    R_uarm_est = np.full((sz_imu, 3, 3), np.nan)
    R_farm_est = np.full((sz_imu, 3, 3), np.nan)
    R_uarm_imu = np.full((sz_imu, 3, 3), np.nan)
    R_farm_imu = np.full((sz_imu, 3, 3), np.nan)

    q_imu_uarm = np.full((4, sz_imu), np.nan)
    q_imu_farm = np.full((4, sz_imu), np.nan)
    
    angle_int = np.full(sz_imu, np.nan)
    angle_est = np.full(sz_imu, np.nan)
    angle_imu = np.full(sz_imu, np.nan)
    
    quat_record = []
    lb = np.array([[-1], [0], [0]]) 
    
    for i in range(sz_imu):
        # pure integration result
        T_se = transFromQuat(int_uarm[:, i:i+1])
        T_es = T_se.T
        R_uarm_int[i, :, :] = T_se
        uarm_int[:, i] = (T_es @ lb).flatten()
        
        T_se = transFromQuat(int_farm[:, i:i+1])
        T_es = T_se.T
        R_farm_int[i, :, :] = T_se
        farm_int[:, i] = (T_es @ lb).flatten()
        
        angle_int[i] = angle_between_vectors(-uarm_int[:, i], farm_int[:, i])
        
        # our method result
        T_se = transFromQuat(qEst_uarm[:, i:i+1])
        T_es = T_se.T
        R_uarm_est[i, :, :] = T_se
        uarm_est[:, i] = (T_es @ lb).flatten()
        
        T_se = transFromQuat(qEst_farm[:, i:i+1])
        T_es = T_se.T
        R_farm_est[i, :, :] = T_se
        farm_est[:, i] = (T_es @ lb).flatten()
        
        angle_est[i] = angle_between_vectors(-uarm_est[:, i], farm_est[:, i])
        
        # pure IMU result
        qimu_uarm_i = qMul(q_ei, quat_uarm[:, idx_imu_start+i:idx_imu_start+i+1])
        T_se = transFromQuat(qimu_uarm_i)
        T_es = T_se.T
        q_imu_uarm[:, i] = qimu_uarm_i.flatten()
        R_uarm_imu[i, :, :] = T_se
        uarm_imu[:, i] = (T_es @ lb).flatten()
        
        qimu_farm_i = qMul(q_ei, quat_farm[:, idx_imu_start+i:idx_imu_start+i+1])
        T_se = transFromQuat(qimu_farm_i)
        quat_record.append(qimu_farm_i)
        T_es = T_se.T
        q_imu_farm[:, i] = qimu_farm_i.flatten()
        R_farm_imu[i, :, :] = T_se
        farm_imu[:, i] = (T_es @ lb).flatten()
        
        angle_imu[i] = angle_between_vectors(-uarm_imu[:, i], farm_imu[:, i])

    # plot forearm xyz
    plt.figure(figsize=(12, 10))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        if i == 0:
            plt.title(f"numParticles: {numParticles}(Farm)")
        
        time_axis = (t_imu_farm[idx_imu_start:idx_imu_start+sz_imu] - t_imu_farm[idx_imu_start]) * 1e-6
        plt.plot(time_axis, farm_imu[i, :], 'r-', label='Xsens DOT')
        plt.plot(time_axis, farm_est[i, :], 'b-', label='Est')
        plt.plot(time_axis, farm_int[i, :], 'm--', label='Int')

        g_start = max(0, ground_offset)
        g_end = min(sz_ground, len(t_image) + ground_offset)
        
        t_start = max(0, -ground_offset)
        t_end = min(len(t_image), sz_ground - ground_offset)
        
        # Ensure length consistency
        length = min(g_end - g_start, t_end - t_start)
        
        if length > 0:
            time_ground = (t_image[t_start:t_start+length] - t_imu_farm[idx_imu_start]) * 1e-6
            plt.plot(time_ground, farm_ground[i, g_start:g_start+length], 'g-', label='Ground Truth')

        plt.ylabel(chr(ord('x') + i))
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    # plot upper arm xyz
    plt.figure(figsize=(12, 10))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        if i == 0:
            plt.title(f"numParticles: {numParticles} (Uarm)")
        
        time_axis = (t_imu_farm[idx_imu_start:idx_imu_start+sz_imu] - t_imu_farm[idx_imu_start]) * 1e-6
        plt.plot(time_axis, uarm_imu[i, :], 'r-', label='Xsens DOT')
        plt.plot(time_axis, uarm_est[i, :], 'b-', label='Est')
        plt.plot(time_axis, uarm_int[i, :], 'm--', label='Int')
        
        g_start = max(0, ground_offset)
        g_end = min(sz_ground, len(t_image) + ground_offset)
        t_start = max(0, -ground_offset)
        t_end = min(len(t_image), sz_ground - ground_offset)
        length = min(g_end - g_start, t_end - t_start)
        
        if length > 0:
            time_ground = (t_image[t_start:t_start+length] - t_imu_farm[idx_imu_start]) * 1e-6
            plt.plot(time_ground, uarm_ground[i, g_start:g_start+length], 'g-', label='Ground Truth')

        plt.ylabel(chr(ord('x') + i))
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    
    # calculate angle error
    print("Calculating angle error...")
    err_angle_int = np.full(sz_ground, np.nan)
    err_angle_imu = np.full(sz_ground, np.nan)
    err_angle_est = np.full(sz_ground, np.nan)
    debug_int = np.full(sz_ground, np.nan)
    debug_imu = np.full(sz_ground, np.nan)
    debug_est = np.full(sz_ground, np.nan)
    debug_ground = np.full(sz_ground, np.nan)
    
    for i in range(len(t_image)):
        t = t_image[i]
        
        i_ground = i + ground_offset
        
        if 0 <= i_ground < sz_ground:
            # Find the nearest IMU time
            i_imu = np.where(t_imu_farm <= t)[0]
            if len(i_imu) > 0:
                idx_imu = i_imu[-1] - idx_imu_start - 1
                if 0 <= idx_imu < sz_imu:
                    # Note: err_angle array length is sz_ground, we use i_ground as the index
                    err_angle_int[i_ground] = angle_int[idx_imu] - angle_ground[i_ground]
                    err_angle_imu[i_ground] = angle_imu[idx_imu] - angle_ground[i_ground]
                    err_angle_est[i_ground] = angle_est[idx_imu] - angle_ground[i_ground]
                    
                    debug_ground[i_ground] = angle_ground[i_ground]
                    debug_int[i_ground] = angle_int[idx_imu]
                    debug_imu[i_ground] = angle_imu[idx_imu]
                    debug_est[i_ground] = angle_est[idx_imu]
    
    plt.figure(figsize=(12, 10))

    # Int vs Ground
    plt.subplot(3, 1, 1)
    mask = ~np.isnan(debug_int) & ~np.isnan(debug_ground)
    idx = np.where(mask)[0]
    plt.plot(idx, debug_int[mask], 'b-', linewidth=1.5, label='Int')
    plt.plot(idx, debug_ground[mask], 'k--', linewidth=1.2, label='Ground')
    plt.xlabel('Time')
    plt.ylabel('debug_int')
    plt.title('Int')
    plt.grid(True)
    plt.legend()

    # IMU vs Ground
    plt.subplot(3, 1, 2)
    mask = ~np.isnan(debug_imu) & ~np.isnan(debug_ground)
    idx = np.where(mask)[0]
    plt.plot(idx, debug_imu[mask], 'r-', linewidth=1.5, label='IMU')
    plt.plot(idx, debug_ground[mask], 'k--', linewidth=1.2, label='Ground')
    plt.xlabel('Time')
    plt.ylabel('debug_imu')
    plt.title('IMU')
    plt.grid(True)
    plt.legend()

    # Est vs Ground
    plt.subplot(3, 1, 3)
    mask = ~np.isnan(debug_est) & ~np.isnan(debug_ground)
    idx = np.where(mask)[0]
    plt.plot(idx, debug_est[mask], 'g-', linewidth=1.5, label='Est')
    plt.plot(idx, debug_ground[mask], 'k--', linewidth=1.2, label='Ground')
    plt.xlabel('Time')
    plt.ylabel('debug_est')
    plt.title('Est')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Calculate and display statistics
    valid_est = ~np.isnan(err_angle_est)
    valid_imu = ~np.isnan(err_angle_imu)
    valid_int = ~np.isnan(err_angle_int)
    
    if np.any(valid_est):
        est_mean = np.mean(np.rad2deg(np.abs(err_angle_est[valid_est])))
        est_std = np.std(np.rad2deg(np.abs(err_angle_est[valid_est])))
        print(f"[Est] Mean: {est_mean:.2f}°, Std: {est_std:.2f}°")
    
    if np.any(valid_imu):
        imu_mean = np.mean(np.rad2deg(np.abs(err_angle_imu[valid_imu])))
        imu_std = np.std(np.rad2deg(np.abs(err_angle_imu[valid_imu])))
        print(f"[IMU] Mean: {imu_mean:.2f}°, Std: {imu_std:.2f}°")
    
    if np.any(valid_int):
        int_mean = np.mean(np.rad2deg(np.abs(err_angle_int[valid_int])))
        int_std = np.std(np.rad2deg(np.abs(err_angle_int[valid_int])))
        print(f"[Int] Mean: {int_mean:.2f}°, Std: {int_std:.2f}°")
    
    # Aggregate into Data structure
    data = Data(
        # Time
        t_imu_us=t_imu_farm[idx_imu_start:idx_imu_start+sz_imu],
        t_image_us=t_image,
        # quaternion results from different methods
        int_uarm=int_uarm,
        int_farm=int_farm,
        est_uarm=qEst_uarm,
        est_farm=qEst_farm,
        imu_uarm=q_imu_uarm,
        imu_farm=q_imu_farm,
        # Rotation matrices (T_se, sensor->earth)
        R_uarm_int=R_uarm_int,
        R_farm_int=R_farm_int,
        R_uarm_est=R_uarm_est,
        R_farm_est=R_farm_est,
        R_uarm_imu=R_uarm_imu,
        R_farm_imu=R_farm_imu,
        # Unit 3d vectors and angles
        uarm_int=uarm_int,
        farm_int=farm_int,
        uarm_est=uarm_est,
        farm_est=farm_est,
        uarm_imu=uarm_imu,
        farm_imu=farm_imu,
        angle_int=angle_int,
        angle_est=angle_est,
        angle_imu=angle_imu,
        # ground truth
        angle_ground=angle_ground,
        uarm_ground=uarm_ground,
        farm_ground=farm_ground,
        # Angular velocity
        w_se_uarm=w_se_uarm,
        w_se_farm=w_se_farm,
        # observations
        kpts_uarm=kpts_uarm,
        kpts_farm=kpts_farm,
        # Calibration
        fx=fx,
        fy=fy,
        T_cw=T_cw,
        q_ie=q_ie,
        q_ei=q_ei,
        # Other config info
        meta={
            "numParticles": numParticles,
            "hz_imu": hz_imu,
            "hz_image": hz_image,
            "idx_sync_w": idx_sync_w,
            "idx_sync_kpts": idx_sync_kpts,
        }
    )
    data.ensure_seconds()

    print("Processing completed!")
    return data

if __name__ == "__main__":
    main()
