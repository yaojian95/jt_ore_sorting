import cv2
import os
from datetime import timedelta

def select_light_position(video_path, frame_number=0):
    """
    交互式选择灯光位置
    返回 (x, y, w, h)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("无法读取指定帧")
    
    # 创建窗口和回调函数
    cv2.namedWindow("Select Light Region")
    roi = {'x':0, 'y':0, 'w':0, 'h':0, 'selecting':False}
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi['x'], roi['y'] = x, y
            roi['w'], roi['h'] = 0, 0
            roi['selecting'] = True
        elif event == cv2.EVENT_MOUSEMOVE and roi['selecting']:
            roi['w'], roi['h'] = x - roi['x'], y - roi['y']
        elif event == cv2.EVENT_LBUTTONUP:
            roi['selecting'] = False
    
    cv2.setMouseCallback("Select Light Region", mouse_callback)
    
    while True:
        display = frame.copy()
        
        # 绘制当前选择的区域
        if roi['w'] > 0 and roi['h'] > 0:
            cv2.rectangle(display, (roi['x'], roi['y']), 
                          (roi['x']+roi['w'], roi['y']+roi['h']), 
                          (0, 255, 0), 2)
        
        cv2.imshow("Select Light Region", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 按Enter键确认选择
            break
        elif key == 27:  # 按ESC键退出
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 确保宽度和高度为正数
    x, y = roi['x'], roi['y']
    w, h = abs(roi['w']), abs(roi['h'])
    
    return (x, y, w, h)

def extract_light_frames(video_path, output_dir, light_position=None, red_threshold=150, frames_around=20):

    if light_position is None:
        print("请选择灯光区域...")
        light_position = select_light_position(video_path)
        print(light_position)
        if light_position is None:
            print("未选择灯光区域，程序退出")
            return
    
    print(light_position)
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 定义灯光检测区域
    x, y, w, h = light_position
    roi = slice(y, y+h), slice(x, x+w)
    
    # 存储需要保存的帧号
    frames_to_save = set()
    
    print("开始分析视频...")
    frame_number = 0
    ore_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 提取红色通道（OpenCV中通道顺序是BGR）
        red_channel = frame[:, :, 2]  # 2对应红色通道
        
        # 检测灯光区域的平均红色亮度
        light_region = red_channel[roi]
        avg_red = light_region.mean()

        # 如果红色亮度超过阈值，记录前后帧
        if avg_red > red_threshold:
            start = max(0, frame_number - frames_around)
            end = min(total_frames, frame_number + frames_around)
            frames_to_save.update(range(start, end+1))

            ore_number += 1
        
        frame_number += 1
        if frame_number % 1000 == 0:
            print(f"已处理 {frame_number}/{total_frames} 帧...")
    
    print(f"分析完成，共找到 {len(frames_to_save)} 帧需要保存")
    # print(f"共找到 {ore_number}")
    
    if not frames_to_save:
        print("未检测到红灯亮的帧，程序退出")
        cap.release()
        return

    # 重新读取视频并保存需要的帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    saved_count = 0
    
    print(frames_to_save)

    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_number in frames_to_save:
            # 计算时间戳
            seconds = frame_number / fps
            timestamp = str(timedelta(seconds=seconds)).replace(":", "-")

            # print(timestamp)
            
            filename = os.path.join(output_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        # if frame_number % 1000 == 0:
        #     print(f"已保存 {saved_count}/{len(frames_to_save)} 帧...")
    
    cap.release()
    print(f"完成！共保存了 {saved_count} 张图片到 {output_dir}")

if __name__ == "__main__":
    video_path = "Video_20250508093153008.avi"  
    output_dir = "./" 
    
    extract_light_frames(video_path, output_dir, light_position=None, red_threshold=60, frames_around=10)