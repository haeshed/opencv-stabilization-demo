import cv2
import datetime
from vidstab.VidStab import VidStab
from vidstab import layer_overlay, layer_blend
import matplotlib.pyplot as plt
import copy

stabilizer = VidStab()
cap = cv2.VideoCapture('vtest2.avi')
# cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
font = cv2.FONT_HERSHEY_DUPLEX
idframe = 0

spar = 1
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * spar
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * spar
while cap.isOpened():
    ret, frame = cap.read()
    idframe = idframe + 1
    if ret:
        text = 'Width: ' + str(w) + ' Height: ' + str(h) + ' frameid=' + str(idframe)
        datet = str(datetime.datetime.now())
        frame = cv2.resize(frame, None, fx=spar, fy=spar, interpolation=cv2.INTER_CUBIC)

        frame2 = copy.copy(frame)
        frame2 = cv2.putText(frame2, text, (10, int(h-20)), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        frame2 = cv2.putText(frame2, datet, (10, 20), font, .5, (255, 255, 255), 1, cv2.LINE_AA)

        window = idframe - 1 if idframe < 31 else 30
        res = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=window, border_type='black', border_size=20)
        text = 'Width: ' + str(w) + ' Height: ' + str(h) + ' frameid=' + str(idframe-1)
        res = cv2.putText(res, text, (10, int(h-5)), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        res = cv2.putText(res, datet, (10, 30), font, .5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('frame', frame2)
        cv2.imshow('stab_frame', res)

    keyboard = cv2.waitKey(1)
    if keyboard == ord('q'):
        break


stabilizer.plot_trajectory()
plt.show()
stabilizer.plot_transforms()
plt.show()

cap.release()
cv2.destroyAllWindows()
