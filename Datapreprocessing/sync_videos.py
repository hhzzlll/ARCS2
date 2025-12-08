import cv2

# 配置视频路径列表
video_paths = [
    r"D:\ARCS2\Project Code\Data\20251204173454\videos\cam1.mp4",
    r"D:\ARCS2\Project Code\Data\20251204173454\videos\cam2.mp4",
    r"D:\ARCS2\Project Code\Data\20251204173454\cam3.mp4"
]

def get_start_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    paused = True
    
    print(f"正在处理: {video_path}")
    print("操作说明:")
    print("  [空格]: 播放/暂停")
    print("  [A] / [D]: 上一帧 / 下一帧 (暂停时)")
    print("  [Enter]: 确认当前帧为起始帧并结束")
    
    window_name = f"Select Start Frame - {video_path}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 540, 960) # 调整窗口大小为 540x960 (原尺寸的一半，适应屏幕)
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            # 如果读完了，回到最后一帧
            current_frame = total_frames - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
        
        # 显示信息
        display_frame = frame.copy()
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(display_frame, f"Frame: {current_frame}/{total_frames} ({status})", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        
        if paused:
            delay = 0
        else:
            delay = 30 # 约 30 FPS
            
        key = cv2.waitKey(delay) & 0xFF
        
        if key == 32: # Space
            paused = not paused
        elif key == 13: # Enter
            break
        elif key == ord('a'):
            current_frame = max(0, current_frame - 1)
            paused = True # 手动步进时暂停
        elif key == ord('d'):
            current_frame = min(total_frames - 1, current_frame + 1)
            paused = True
            
        if not paused:
            current_frame += 1
            if current_frame >= total_frames:
                current_frame = total_frames - 1
                paused = True

    cap.release()
    cv2.destroyAllWindows()
    return current_frame

def main():
    # 1. 交互式选择每个视频的起始帧
    start_frames = []
    for path in video_paths:
        start_frame = get_start_frame(path)
        start_frames.append(start_frame)
        print(f"视频 {path} 的起始帧选定为: {start_frame}")
        
    # 2. 裁剪并保存视频
    for i, path in enumerate(video_paths):
        start_f = start_frames[i]
        output_path = path.replace(".mp4", "_synced.mp4")
        
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"正在导出: {output_path} (从第 {start_f} 帧开始)...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            count += 1
            if count % 100 == 0:
                print(f"已处理 {count} 帧", end='\r')
                
        cap.release()
        out.release()
        print(f"\n完成: {output_path}")

if __name__ == "__main__":
    main()
