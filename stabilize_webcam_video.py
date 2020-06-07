import cv2
import datetime
import numpy as np
from vidstab.VidStab import VidStab
import matplotlib.pyplot as plt
import copy

# initialize stabilizer
stabilizer = VidStab()

# sets video capture source, live feed or using existing file
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('vtest2.avi')

# set codec & font, initialize frame counter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
font = cv2.FONT_HERSHEY_DUPLEX
idframe = 0

# allows user to set parameter for size of frame & initializes new frame size
spar = .8
bor = 20
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * spar
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * spar

out = cv2.VideoWriter('output.avi',fourcc, 30.0, (int(2*(w+2*bor)), int(h+2*bor)))

while cap.isOpened():
    ret, frame = cap.read()
    idframe = idframe + 1
    # check if frame valid
    if ret:
        # defines text over raw frame & resize frame w/user set parameter
        text = 'Width: ' + str(w) + ' Height: ' + str(h) + ' frameid=' + str(idframe)
        datet = str(datetime.datetime.now())
        frame = cv2.resize(frame, None, fx=spar, fy=spar, interpolation=cv2.INTER_CUBIC)
        # duplicates frame to avoid double text on stabilized frame
        frame2 = copy.copy(frame)
        frame2 = cv2.putText(frame2, text, (10, int(h-bor)), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        frame2 = cv2.putText(frame2, datet, (10, bor), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        frame2 = cv2.copyMakeBorder(frame2, bor, bor, bor, bor, borderType=0)
        # initiates stabilization parameter for first few frames, enables smooth stabilization from start
        window = idframe - 1 if idframe < 31 else 30
        # stabilization function based on parameters given forehand
        res = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=window, border_type='black', border_size=bor)
        # text addition (date, time, frame number) for stabilized frame
        text = 'Width: ' + str(w) + ' Height: ' + str(h) + ' frameid=' + str(idframe-1)
        res = cv2.putText(res, text, (10, int(h+5)), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        res = cv2.putText(res, datet, (10, 30), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        # res = res[int(h/8):int(h), int(w/8):int(w)]

        # opens 2 individual windows containing frames (videos) of raw & stabilized
        # cv2.imshow('frame', frame2)
        # cv2.imshow('stab_frame', res)

        # concatenates the 2 windows to a single window side by side
        concat = np.concatenate((frame2, res), axis=1)
        cv2.imshow('stabilization comparison', concat)
        out.write(concat)

    # waits for optional 'q' or 'esc' user keystroke to quit at any time
    keyboard = cv2.waitKey(1)
    if keyboard == ord('q') or keyboard == 27: break
    # option to pause video using spacebar, continue on any key
    if keyboard == 32: cv2.waitKey(0)
    # auto-exits when video is over
    if idframe == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)): break

# shows graphic representation of video stabilization trajectory, option for exporting data
stabilizer.plot_trajectory()
plt.show()
stabilizer.plot_transforms()
plt.show()
# auto-release video capture & deletes open windows
cap.release()
out.release()
cv2.destroyAllWindows()
