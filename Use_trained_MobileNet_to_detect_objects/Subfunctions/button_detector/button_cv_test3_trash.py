import numpy as np
import cv2
import math

img = cv2.imread('image.bmp')

# Prepocess
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
flag, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Find contours
img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Select long perimeters only
perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
numcards=len(listindex)

card_number = -1 #just so happened that this is the worst case
stencil = np.zeros(img.shape).astype(img.dtype)
cv2.drawContours(stencil, [contours[listindex[card_number]]], 0, (255, 255, 255), cv2.FILLED)
res = cv2.bitwise_and(img, stencil)
cv2.imwrite("out.bmp", res)
canny = cv2.Canny(res, 100, 200)
cv2.imwrite("canny.bmp", canny)

import cv2
import numpy as np

img = cv2.imread('sof.jpg')
img = cv2.resize(img,(500,500))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray,127,255,0)
contours,hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>5000:  # remove small areas like noise etc
        hull = cv2.convexHull(cnt)    # find the convex hull of contour
        hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
        if len(hull)==4:
            cv2.drawContours(img,[hull],0,(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
