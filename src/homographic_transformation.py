import cv2 as cv
import numpy as np

input_video = r'C:\Users\gokul\Desktop\Computer_vision_mini_project\data\test_video_1.mp4'

vid = cv.VideoCapture(input_video)
check = 0
while vid.isOpened():

    ret,frame = vid.read()
    #frame shape : (850, 478, 3)
    if not ret:
        exit()

    #dummy points
    # tl = [80,300]
    # bl = [239,480]
    # tr = [240,300]
    # br = [478,440]
    tl = [40,380]
    bl = [80,450]
    tr = [400,320]
    br = [478,390]

    cv.circle(frame,tl,5,[0,255,0],-1)
    cv.circle(frame,tr,5,[0,255,0],-1)
    cv.circle(frame,bl,5,[0,255,0],-1)
    cv.circle(frame,br,5,[0,255,0],-1)

    src_pts = np.array([tl, bl, tr, br], dtype=np.float32)
    # dst_pts = np.array([[[0, 0], [0, 640], [480, 0], [480, 640]]], dtype=np.float32)
    dst_pts = np.array([[[0, 0], [0, 480], [640, 0], [640, 480]]], dtype=np.float32)

    perspective_mat = cv.getPerspectiveTransform(src_pts, dst_pts)

    trans_frame = cv.warpPerspective(frame, perspective_mat, (640,480))

    cv.imshow('Original video', frame)
    cv.imshow('Transformed fisheeye', trans_frame)
    
    check = check + 1

    if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()