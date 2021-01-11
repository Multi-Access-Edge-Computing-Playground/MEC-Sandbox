import cv2
import numpy as np
import sys

img = cv2.imread(str("rectanglepicture.png"),0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh,1,2)



for contour in contours:
    area = cv2.contourArea(contour)
    if area>100000:
        contours.remove(contour)




cnt = contours[0]

epsilon = 0.02*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

print("No of rectangles")
print(len(approx))


#finding the centre of the contour
M = cv2.moments(cnt)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

print (cx)
print (cy)
