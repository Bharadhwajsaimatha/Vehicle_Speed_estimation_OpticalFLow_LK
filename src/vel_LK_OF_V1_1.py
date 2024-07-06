'''
Importing necessary libraries

'''
import numpy as np
import argparse
import cv2 as cv
from pathlib import Path
import os
import time


'''
Neccesary file paths    
'''
curr_dir = os.getcwd()
dir_path = Path(curr_dir)
default_video = dir_path/'data/cars_on_highway.mp4'
default_op_video = dir_path/'output/output_OF_LK_video.mp4'

lk_params = dict(winSize=(50, 50),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 100, 
                       qualityLevel = 0.3, 
                       minDistance = 7, 
                       blockSize = 5 ) 


#ToDo
METER_PER_PIXEL = 0.1

def time_decorator(func) -> None:
    def time_wrapper(*args,**kwargs):
        start_time = time.perf_counter()
        print('Execution started!')
        func()
        end_time = time.perf_counter()
        print(f'Execution completed in {end_time-start_time} seconds.')
    return time_wrapper

def arg_parser() -> argparse:
    parser = argparse.ArgumentParser(description='input video file')
    parser.add_argument('--input_file',
                        type=str,
                        default=default_video,
                        help='input video file')
    
    parser.add_argument('--webcam',
                        type=int,
                        default=0,
                        help='use webcam instead?')
    
    parser.add_argument('--output_file_name',
                        type=str,
                        default=default_op_video,
                        help='Name of the output video')
    
    parser.add_argument('--transform',
                        type = bool,
                        default = False,
                        help='Do you need perspective transform?')
    
    args = parser.parse_args()
    return args
    
def transform_frame(frame)->np.array:
    #Window for transformation
    # tl = [40,380]
    # bl = [80,500]
    # tr = [400,320]
    # br = [478,440]
    tl = [40,380]
    bl = [80,450]
    tr = [400,320]
    br = [478,390]
    src_pts = np.array([tl, bl, tr, br], dtype=np.float32)
    dst_pts = np.array([[[0, 0], [0, 640], [480, 0], [480, 640]]], dtype=np.float32)
    perspective_mat = cv.getPerspectiveTransform(src_pts, dst_pts)
    trans_frame = cv.warpPerspective(frame, perspective_mat, (640,480))

    return trans_frame

def calculate_speed(prev_pts, curr_pts, fps, meters_per_pixel):
    displacement = np.linalg.norm(curr_pts - prev_pts, axis=1)
    filtered_displacement = [pt for pt in displacement if pt > 2 and pt < 7]
    if len(filtered_displacement) == 0:
        filtered_displacement = 0
    speed = np.mean(filtered_displacement) * fps * meters_per_pixel * 3.6
    return speed

def get_fps(vid):
    ip_fps = vid.get(cv.CAP_PROP_FPS)
    if ip_fps == 0:
        ip_fps = 30
    return ip_fps

def calculate_optical_flow(vid,fps,transform_flag):
    status , old_org_frame = vid.read()
    if transform_flag == True:
        old_frame = transform_frame(old_org_frame)
    else:
        old_frame = old_org_frame
    old_gray = cv.cvtColor(old_frame,cv.COLOR_BGR2GRAY)

    while vid.isOpened():
        status , org_frame = vid.read()
        if not status:
            print('Error in capturing frames.and/or Video read complete.\nExiting.......')
            exit()
        if transform_flag == True:
            frame = transform_frame(org_frame)
        else:
            frame = org_frame
        gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)
        prev_pts = cv.goodFeaturesToTrack(old_gray, mask = None, 
                             **feature_params)
        
        curr_pts, st, err = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, prev_pts, None, **lk_params)

        good_new = curr_pts[st == 1]
        good_old = prev_pts[st == 1]

        average_speed = calculate_speed(good_old, good_new, fps, METER_PER_PIXEL)
        speed_text = f"Average speed: {average_speed:.2f} km/hr"
        cv.putText(org_frame, speed_text, (15, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            # frame = cv.circle(frame, (a, b), 5, (0, 0, 255), -1)

        # op_frame = cv.add(frame,mask)
        time.sleep(0.022)
        cv.imshow('Output_video', org_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        old_gray = gray_frame.copy()
        prev_pts = good_new.reshape(-1, 1, 2)
        

@time_decorator
def main() -> None:
    args = arg_parser()

    if args.webcam == 1:
        vid = cv.VideoCapture(0)
        if not vid.isOpened():
            print('Couldn\'t open webcam')
            print('Stopping execution')
            exit()
    else:
        vid = cv.VideoCapture(str(args.input_file))
        if not vid.isOpened():
            print('Couldn\'t open the provided video file. Make sure it is a video file compatible with opencv-python')
            print('Stopping execution')
            exit()

        ip_fps = get_fps(vid)
        print(f'Meter per pixel is set to : {METER_PER_PIXEL}\nPlease change this values based on the scene')
        if args.transform == True:
            print('Warning! Perspective transformation applied')
        calculate_optical_flow(vid,ip_fps, args.transform)
        vid.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()


