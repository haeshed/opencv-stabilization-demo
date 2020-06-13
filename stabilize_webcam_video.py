import copy
import cv2
from datetime import datetime
import numpy as np
from vidstab.VidStab import VidStab
import matplotlib.pyplot as plt
import os


def main():
    # initialize stabilizer
    stabilizer = VidStab()
    # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    # sets video capture source, live feed or using existing file
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('vtest2.avi')
    # cap = cv2.VideoCapture("autovideosrc ! decodebin ! appsink", cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture("autovideosrc ! decodebin ! appsink", cv2.CAP_FFMPEG)
    # cap = cv2.VideoCapture("gst-launch-1.0 appsink ! application/x-rtp, encoding-name=H264,payload=96 !  rtph264depay ! h264parse ! avdec_h264 !  autovideosink", cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture('videotestsrc ! video/x-raw,framerate=20/1 ! videoscale ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture('autovideosrc ! video/x-raw,framerate=20/1 ! videoscale ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture('udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink',cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture('autovideosrc ! decodebin ! videorate ! video/x-raw,framerate=20/1 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture('autovideosrc ! decodebin ! appsink', cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture('autovideosrc ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture('rtsp://127.0.0.1:5000/out.h264')
    # cap = cv2.VideoCapture('autovideosrc ! ffmpegcolorspace ! video/x-raw-rgb ! appsink', cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture('gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=H264,payload=96 !  rtph264depay ! h264parse ! avdec_h264 !  autovideosink')

    # set codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # allows user to set parameter for size of frame & initializes new frame size
    spar = .7
    bor = 20
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * spar
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * spar
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(2 * (w + 2 * bor)), int(h + 2 * bor)))

    idframe = 0
    while cap.isOpened():
        ret, frame = cap.read()
        idframe += 1

        # check if frame valid
        if ret:
            # defines text over raw frame & resize frame w/user set parameter
            text = f'Width: {w} Height: {h} frame id: {idframe}'
            datet = str(datetime.now())
            frame = cv2.resize(frame, None, fx=spar, fy=spar, interpolation=cv2.INTER_CUBIC)

            # duplicates frame to avoid double text on stabilized frame
            frame_with_text = copy.copy(frame)
            frame_with_text = cv2.putText(frame_with_text, text, (10, int(h - bor)), cv2.FONT_HERSHEY_DUPLEX, .5,
                                          (255, 255, 255), 1, cv2.LINE_AA)
            frame_with_text = cv2.putText(frame_with_text, datet, (10, bor), cv2.FONT_HERSHEY_DUPLEX, .5,
                                          (255, 255, 255), 1, cv2.LINE_AA)
            frame_with_text = cv2.copyMakeBorder(frame_with_text, bor, bor, bor, bor, borderType=0)

            # initiates stabilization parameter for first few frames, enables smooth stabilization from start
            window = idframe - 1 if idframe < 31 else 30
            # stabilization function based on parameters given forehand
            res = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=window, border_type='black',
                                             border_size=bor)

            # text addition (date, time, frame number) for stabilized frame
            text = f'Width: {w} Height: {h} frame id: {idframe - 1}'
            res = cv2.putText(res, text, (10, int(h + 5)), cv2.FONT_HERSHEY_DUPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            res = cv2.putText(res, datet, (10, 30), cv2.FONT_HERSHEY_DUPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)

            # concatenates the 2 windows to a single window side by side
            concat = np.concatenate((frame_with_text, res), axis=1)
            cv2.imshow('stabilization comparison', concat)
            out.write(concat)

        # waits for optional 'q' or 'esc' user keystroke to quit at any time
        keyboard = cv2.waitKey(1)
        if keyboard == ord('q') or keyboard == 27:
            break
        # option to pause video using spacebar, continue on any key
        elif keyboard == 32:
            cv2.waitKey(0)
        # auto-exits when video is over
        if idframe == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break

    # shows graphic representation of video stabilization trajectory, option for exporting data
    stabilizer.plot_trajectory()
    plt.show()
    stabilizer.plot_transforms()
    plt.show()
    # auto-release video capture & deletes open windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
