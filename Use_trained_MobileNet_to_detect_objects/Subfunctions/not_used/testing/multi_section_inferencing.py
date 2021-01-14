import cv2
import numpy as np
import time
from imutils.video import WebcamVideoStream

cap = WebcamVideoStream(0).start()
time.sleep(0.5)
cv2_im = cap.read()
(h, w) = cv2_im.shape[:2]
print(h, "/", w)
# out = cv2.VideoWriter('./detection_%s.avi'%(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
# out = cv2.VideoWriter('./detection_.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
divider = 3
while True:
    cv2_im = cap.read()
    (h, w) = cv2_im.shape[:2]
    x_step = int(w/divider)
    y_step = int(h/divider)
    # outer
    for x in range(divider):
        for y in range(divider):
            pt1 = x*x_step,y*y_step
            pt2 = (x+1)*x_step,(y+1)*y_step
            cv2.rectangle(cv2_im,pt1,pt2,(0,0,255),4)
    # inner
    for x in range(divider-1):
        for y in range(divider-1):
            pt1 = int(x*x_step*divider/2),int(y*y_step*divider/2)
            pt2 = int((x+1)*x_step*divider/2),int((y+1)*y_step*divider/2)
            cv2.rectangle(cv2_im,pt1,pt2,(0,255,0),3)
    # pt1 = int(0.5*x_step),int(0.5*y_step)
    # pt2 = int(1.5*x_step),int(1.5*y_step)
    # cv2.rectangle(cv2_im,pt1,pt2,(0,255,0),2)
    # middle
    pt1 = int(w/2-0.5*x_step),int(h/2-0.5*y_step)
    pt2 = int(w/2+0.5*x_step),int(h/2+0.5*y_step)
    cv2.rectangle(cv2_im,pt1,pt2,(255,0,0),2)
    cv2.imshow("Camera Feed", cv2_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
# out.release()
cv2.destroyWindow(str(0))
