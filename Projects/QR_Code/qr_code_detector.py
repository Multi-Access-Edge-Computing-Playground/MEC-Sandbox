import cv2
import numpy as np
from Subfunctions.WebcamVideoStream import WebcamVideoStream
from pyimagesearch.transform import *
import time
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
    decodedText, points, _ = qrCodeDetector.detectAndDecode(image)




    if points is not None:
        #print(points)
        nrOfPoints = len(points[0])

        for i0 in range(nrOfPoints):
            nextPointIndex = (i0+1) % nrOfPoints
            cv2.line(image, tuple(points[0][i0]), tuple(points[0][nextPointIndex]), (255,0,0), 5)
            cv2.circle(image,tuple(points[0][i0]), 5, (0,0,255), 5)

        print(decodedText)

    #shows the WebcamVideoStream
    t1=time.time()
    frame_count+=1
    if t1-t0>=1.0:
        framerate=frame_count
        frame_count=0
        t0=time.time()
    cv2.putText(image, "Framerate: "+str(framerate),
                (10,15),fontStyle, 0.5, (255,255,255), 2)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
