from ultralytics import YOLO
import cv2 as cv
import time
import csv
import os
from pathlib import Path
import toml
from tqdm import tqdm

# hard-coded joint pair
KEYPOINT_DICT={
    "nose":0,
    "left_eye":1,
    "left_ear":2,
    "right_eye":3,
    "right_ear":4,
    "left_shoulder":5,
    "right_shoulder":6,
    "left_elbow":7,
    "right_elbow":8,
    "left_wrist":9,
    "right_wrist":10,
    "left_hip":11,
    "right_hip":12,
    "left_knee":13,
    "right_knee":14,
    "left_ankle":15,
    "right_ankle":16
}

KEYPOINT_PAIR = [(KEYPOINT_DICT["left_shoulder"],KEYPOINT_DICT["left_elbow"]),
                 (KEYPOINT_DICT["left_elbow"],KEYPOINT_DICT["left_wrist"]),
                 (KEYPOINT_DICT["right_shoulder"],KEYPOINT_DICT["right_elbow"]),
                 (KEYPOINT_DICT["right_elbow"],KEYPOINT_DICT["right_wrist"]),
                 (KEYPOINT_DICT["left_hip"],KEYPOINT_DICT["left_knee"]),
                 (KEYPOINT_DICT["left_knee"],KEYPOINT_DICT["left_ankle"]),
                 (KEYPOINT_DICT["right_hip"],KEYPOINT_DICT["right_knee"]),
                 (KEYPOINT_DICT["right_knee"],KEYPOINT_DICT["right_ankle"]),
                 (KEYPOINT_DICT["left_shoulder"],KEYPOINT_DICT["right_shoulder"]),
                 (KEYPOINT_DICT["left_hip"],KEYPOINT_DICT["right_hip"]),
                 (KEYPOINT_DICT["left_shoulder"],KEYPOINT_DICT["left_hip"]),
                 (KEYPOINT_DICT["right_shoulder"],KEYPOINT_DICT["right_hip"])]


# Load a model
model = YOLO('Datapreprocessing\\yolov8n-pose.pt')  # build from YAML and transfer weights

# Read project_name from config/app_config.toml and build paths under ./Data/{project_name}
project_root = Path(__file__).resolve().parent.parent  # .../Project Code
cfg_path = project_root / 'config' / 'config.toml'
cfg_data = toml.load(str(cfg_path))
project_name = cfg_data['project']['project_name']

# used for recording
fcount = 0
t_prev = 0
FPS = 60            # recording frame rate
TIME_LAPSE = 1/FPS
THRESHOLD = 0.75

video_path = project_root / 'Data' / project_name / 'videos' / 'cam01.mp4'
# video_path = project_root / 'Data' / project_name / 'cam3.mp4'
cap = cv.VideoCapture(str(video_path))
fps = cap.get(cv.CAP_PROP_FPS)
print(fps)

if not cap.isOpened():
    print("Cannot open file")
    exit()

# Setup VideoWriter
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
output_video_path = video_path.parent / f"{video_path.stem}_pose.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out_video = cv.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames)

out = []
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break

    results = model(frame, show=False, verbose=False)
    
    # Save detected frame
    annotated_frame = results[0].plot()
    out_video.write(annotated_frame)

    keypoints= results[0].keypoints.xy      # all keypoints in original pixel coordinate for each detected person
    score = results[0].keypoints.conf
    
    # no person detected
    if (keypoints is None) or (score is None):
        continue
    
    # 只处理置信度最高的一个人
    if len(keypoints) > 0:
        # 计算每个人所有关键点的平均置信度
        mean_scores = score.mean(dim=1)
        best_person_idx = mean_scores.argmax().item()
        
        # 提取最佳匹配的人
        person = keypoints[best_person_idx]
        conf = score[best_person_idx]
        
        kps = person.tolist()
        val  = conf.tolist()
        # keypoints visualization 
        # for pair in KEYPOINT_PAIR:
        #     try:
        #         if val[pair[0]]<THRESHOLD or val[pair[1]]<THRESHOLD:
        #             continue
        #         pt1 = tuple(int(i) for i in kps[pair[0]])
        #         pt2 = tuple(int(i) for i in kps[pair[1]])
        #         frame = cv.circle(frame, pt1, 6, (255,255,255) , -1) #(BGR)
        #         frame = cv.circle(frame, pt2, 6, (255,255,255) , -1)
        #         frame = cv.line(frame, pt1, pt2, (0,255,0), 3)
        #     except:
        #         continue

        #only output right_elbow-right_wrist
        right_shoulder = list(int(i) for i in kps[KEYPOINT_DICT["right_shoulder"]])
        right_elbow = list(int(i) for i in kps[KEYPOINT_DICT["right_elbow"]])
        right_wrist = list(int(i) for i in kps[KEYPOINT_DICT["right_wrist"]])
        right_shoulder_conf = val[KEYPOINT_DICT["right_shoulder"]]
        right_elbow_conf = val[KEYPOINT_DICT["right_elbow"]]
        right_wrist_conf = val[KEYPOINT_DICT["right_wrist"]]

        right_hip = list(int(i) for i in kps[KEYPOINT_DICT["right_hip"]])
        right_knee = list(int(i) for i in kps[KEYPOINT_DICT["right_knee"]])
        right_ankle = list(int(i) for i in kps[KEYPOINT_DICT["right_ankle"]])
        right_hip_conf = val[KEYPOINT_DICT["right_hip"]]
        right_knee_conf = val[KEYPOINT_DICT["right_knee"]]
        right_ankle_conf = val[KEYPOINT_DICT["right_ankle"]]
        
        # add to original frame for visualization
        #frame = cv.circle(frame, right_elbow, 6, (255,255,255) , -1) #(BGR)
        #frame = cv.circle(frame, right_wrist, 6, (255,255,255) , -1)
        #frame = cv.line(frame, right_elbow, right_wrist, (0,255,0), 3)
        
        out.append([fcount] + 
                   right_shoulder + [right_shoulder_conf] + 
                   right_elbow + [right_elbow_conf] + 
                   right_wrist + [right_wrist_conf] +
                   right_hip + [right_hip_conf] +
                   right_knee + [right_knee_conf] +
                   right_ankle + [right_ankle_conf])

    #cv.imwrite(f"./{fcount}.jpg",frame)
    fcount+=1
    pbar.update(1)

# When everything done, release the capture
pbar.close()
cap.release()
out_video.release()
cv.destroyAllWindows()

# Write measurement CSV into ./Data/{project_name}/measurement_data
measurement_dir = project_root / 'Data' / project_name / 'measurement_data'
os.makedirs(measurement_dir, exist_ok=True)
with open(str(measurement_dir / 'measurement.csv'), "w", newline="") as f:
    header = ['index', 
              'rshoulder u','rshoulder v', 'rshoulder conf',
              'relbow u', 'relbow v', 'relbow conf',
              'rwrist u','rwrist v','rwrist conf',
              'rhip u', 'rhip v', 'rhip conf',
              'rknee u', 'rknee v', 'rknee conf',
              'rankle u', 'rankle v', 'rankle conf']
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(out)
    f.close()







