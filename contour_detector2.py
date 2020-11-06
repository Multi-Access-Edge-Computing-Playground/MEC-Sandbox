import cv2
import numpy as np
from Subfunctions.WebcamVideoStream import WebcamVideoStream
import time
import sys
from scipy.spatial.transform import Rotation as R
cap = WebcamVideoStream(0).start()
time.sleep(0.5)
frame_count=0
framerate=0
qrCodeDetector = cv2.QRCodeDetector()
fontStyle = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
fontThickness = 2
t0=time.time()
while True:

    image = cap.read()
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100000:
            contours.remove(contour)

    cnt = contours[0]

    epsilon = 0.02*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    print("No. of rectanlges: ", len(approx))

    M = cv.moments(cnt)


    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    print (cx,cy)






    t1=time.time()
    frame_count+=1
    if t1-t0>=1.0:
        framerate=frame_count
        frame_count=0
        t0=time.time()
    cv2.putText(image, "Framerate: "+str(framerate),
                (10,15),fontStyle, 0.5, (0,0,0), 2)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
