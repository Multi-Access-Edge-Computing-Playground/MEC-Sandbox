from __future__ import print_function
from __future__ import division
from Subfunctions.WebcamVideoStream import WebcamVideoStream
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi, degrees
cap = cv.VideoCapture(0)
cap.set(10,100)
kernel = np.ones((5,5),np.uint8)
print("Press 'q' on your Keyboard to exit this program.")
#puts together picture and edited pictures and webcam and edited webcam
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
#draws the axis
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    #angle as radiant
    angle = atan2(p[1] - q[1], p[0] - q[0])
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    #Pfeil wird um einen Skalierungsfaktor verlängert
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
#finds out angle, centre and
def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    mean = np.empty((0))
    mean, eigenvektoren, eigenwerte = cv.PCACompute2(data_pts, mean)
    centre = (int(mean[0,0]), int(mean[0,1]))
    #creates a yellow point in the centre of the rectangles
    cv.circle(img, centre, 3, (0, 255, 255), 2)
    p1 = (centre[0] + 0.02 * eigenvektoren[0,0] * eigenwerte[0,0], centre[1] + 0.02 * eigenvektoren[0,1] * eigenwerte[0,0])
    p2 = (centre[0] - 0.02 * eigenvektoren[1,0] * eigenwerte[1,0], centre[1] - 0.02 * eigenvektoren[1,1] * eigenwerte[1,0])
    drawAxis(img, centre, p1, (0, 0, 255), 1)
    drawAxis(img, centre, p2, (255, 255, 0), 5)
    #calculates each rectangles angle in radiant and degree
    angle = atan2(eigenvektoren[0,1], eigenvektoren[0,0])
    angleDegree = degrees(angle)
    angleInt = int(angleDegree)
    angleString = str(angleInt)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, angleString, centre, font, 1, (200,255,155), 2, cv.LINE_AA)
    return angle
#webcam
while True:
    success, image = cap.read()
    imgGray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv.Canny(imgBlur,50,50)
    imgZero = np.zeros_like(image)
    imgContour = image.copy()
    _, bw = cv.threshold(imgCanny, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        if area < 1e2 or 1e5 < area:
            continue
        cv.drawContours(imgContour, contours, i, (255, 0, 0), 2)
        getOrientation(c, imgContour)
    imgStack = stackImages(0.6,([image,imgGray,imgBlur],
                                [imgCanny,imgContour,imgZero]))
    #Webcam show
    cv.imshow("Stacked Images Webcam", imgStack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
