import cv2
import numpy as np
# from Subfunctions.WebcamVideoStream import WebcamVideoStream
import time
import sys
from scipy.spatial.transform import Rotation as R
camera_id = 0 #Change this ID if camera images can't be read or wrong camera was used.
cap = cv2.VideoCapture(camera_id)
time.sleep(0.5) #heat up the Sensor
frame_count=0
framerate=0
fontStyle = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
fontThickness = 2
t0=time.time()
while True:
    image = cap.read()

    #Insert Code here to calculate the angle of a rectangle
    # maybe get results from here:
    # https://stackoverflow.com/questions/34237253/detect-centre-and-angle-of-rectangles-in-an-image-using-opencv

    #Calculate and display the frame rate
    t1=time.time()
    frame_count+=1
    if t1-t0>=1.0:
        framerate=frame_count
        frame_count=0
        t0=time.time()
    cv2.putText(image, "Framerate: "+str(framerate),
                (10,15),fontStyle, 0.5, (255,255,255), 2)

    #Display the image
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #breaks out of the while loop if "q" is typed
cv2.destroyAllWindows()
