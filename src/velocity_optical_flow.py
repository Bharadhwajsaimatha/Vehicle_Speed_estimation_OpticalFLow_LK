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
lbl_file = dir_path/'yolo_v4/coco.names'
cfg_file = dir_path/'yolo_v4/yolov4.cfg'
weight_file = dir_path/'yolo_v4/yolov4.weights'
default_video = dir_path/'data/cars_on_highway.mp4'
default_op_video = dir_path/'output/output_OF_LK_video.mp4'

lk_params = dict(winSize=(100, 100),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 100, 
                       qualityLevel = 0.3, 
                       minDistance = 7, 
                       blockSize = 7 ) 

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
    
    args = parser.parse_args()
    return args

def calculate_speed(prev_pts, curr_pts, fps, meters_per_pixel):
    displacement = np.linalg.norm(curr_pts - prev_pts, axis=1)
    filtered_displacement = [pt for pt in displacement if pt > 1]
    if len(filtered_displacement) == 0:
        filtered_displacement = 0
    speed = np.mean(filtered_displacement) * fps * meters_per_pixel * 3.6
    return speed

def get_fps(vid):
    ip_fps = vid.get(cv.CAP_PROP_FPS)
    if ip_fps == 0:
        ip_fps = 30
    return ip_fps

def initialize_video_writer(vid,fps, op_file):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv.VideoWriter(str(op_file), fourcc, fps, (width, height))
    return video_writer

def calculate_optical_flow(vid,fps):
    #video_writer
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    # detection_model  = cv.dnn.readNet(str(weight_file), str(cfg_file))
    # layer_names = detection_model.getLayerNames()
    # output_layers = [layer_names[i - 1] for i in detection_model.getUnconnectedOutLayers()]
    # classes = []
    # prev_pts = None
    # old_gray = None
    status , old_frame = vid.read()
    old_gray = cv.cvtColor(old_frame,cv.COLOR_BGR2GRAY)
    # dummy = 0
    # mask = np.zeros_like(old_frame)

    while vid.isOpened():
        status , frame = vid.read()
        if not status:
            print('Error in capturing frames.and/or Video read complete.\nExiting.......')
            exit()
        gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)
        prev_pts = cv.goodFeaturesToTrack(old_gray, mask = None, 
                             **feature_params) 
        #YOLO part
        # with open(lbl_file, "r") as f:
        #     classes = [line.strip() for line in f.readlines()]
        # blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # detection_model.setInput(blob)
        # yolo_outs = detection_model.forward(output_layers)
        # class_ids = []
        # confidences = []
        # boxes = []
        # for out in yolo_outs:
        #     for detection in out:
        #         scores = detection[5:]
        #         class_id = np.argmax(scores)
        #         confidence = scores[class_id]
        #         if confidence > 0.5 and classes[class_id] == "car":
        #             center_x = int(detection[0] * width)
        #             center_y = int(detection[1] * height)
        #             w = int(detection[2] * width)
        #             h = int(detection[3] * height)
        #             x = int(center_x - w / 2)
        #             y = int(center_y - h / 2)
        #             boxes.append([x, y, w, h])
        #             confidences.append(float(confidence))
        #             class_ids.append(class_id)
        # indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # new_pts = []
        # for i in range(len(boxes)):
        #     if i in indexes:
        #         x, y, w, h = boxes[i]
        #         new_pts.append([x + w / 2, y + h / 2])
        #         frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # new_pts = []

        # if prev_pts is None:
        #     prev_pts = np.array(new_pts, dtype=np.float32).reshape(-1, 1, 2)
        #     old_gray = gray_frame.copy()
        #     continue
        
        curr_pts, st, err = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, prev_pts, None, **lk_params)

        # if curr_pts is None:
        #     prev_pts = np.array(new_pts, dtype=np.float32).reshape(-1, 1, 2)
        #     old_gray = gray_frame.copy()
        #     continue

        good_new = curr_pts[st == 1]
        good_old = prev_pts[st == 1]

        # if dummy < 2:
        #     print(f'Good new: {good_new}')
        #     print(f'old points:{good_old}')

        average_speed = calculate_speed(good_old, good_new, fps, METER_PER_PIXEL)
        speed_text = f"Average speed: {average_speed:.2f} km/hr"
        cv.putText(frame, speed_text, (15, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv.circle(frame, (a, b), 5, (0, 0, 255), -1)

        # op_frame = cv.add(frame,mask)
        cv.imshow('Output_video', frame)
        # video_writer.write(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        old_gray = gray_frame.copy()
        prev_pts = good_new.reshape(-1, 1, 2)
        # dummy += 1
        

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
        # vid_writer = initialize_video_writer(vid,ip_fps,args.output_file_name)
        # print(f"VideoWriter initialized: {vid_writer.isOpened()}")

        # calculate_optical_flow(vid,ip_fps,vid_writer)
        calculate_optical_flow(vid,ip_fps)
        vid.release()
        # vid_writer.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()


