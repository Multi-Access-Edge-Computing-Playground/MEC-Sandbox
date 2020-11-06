from __future__ import print_function
from __future__ import division
from Subfunctions.WebcamVideoStream import WebcamVideoStream
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi, degrees


cap = WebcamVideoStream(0).start()


#zeichnet die Pfeile, welche im rechten Winkel aufeinander stehen
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)


    #Winkel als radiant
    angle = atan2(p[1] - q[1], p[0] - q[0])
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    #Pfeil wird um einen Skalierungsfaktor verlängert
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)



def getOrientation(pts, img):

    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]


    mean = np.empty((0))
    mean, eigenvektoren, eigenwerte = cv.PCACompute2(data_pts, mean)
    #Speichert das Centre der Rechtecke
    centre = (int(mean[0,0]), int(mean[0,1]))


    #erstellt an der Mitte der Rechtecke einen gelben Punkt
    cv.circle(img, centre, 3, (0, 255, 255), 2)
    p1 = (centre[0] + 0.02 * eigenvektoren[0,0] * eigenwerte[0,0], centre[1] + 0.02 * eigenvektoren[0,1] * eigenwerte[0,0])
    p2 = (centre[0] - 0.02 * eigenvektoren[1,0] * eigenwerte[1,0], centre[1] - 0.02 * eigenvektoren[1,1] * eigenwerte[1,0])
    drawAxis(img, centre, p1, (0, 0, 255), 1)
    drawAxis(img, centre, p2, (255, 255, 0), 5)
    #Orientierung am Winkel in radiant
    angle = atan2(eigenvektoren[0,1], eigenvektoren[0,0])
    #Output: Winkel in Degree
    print(degrees(angle))
    return angle
parser = argparse.ArgumentParser(description='Code for Introduction to Principal Component Analysis (PCA) tutorial.\
                                              This program demonstrates how to use OpenCV PCA to extract the orientation of an object.')
parser.add_argument('--input', help='Path to input image.', default='pca_test1.jpg')
args = parser.parse_args()

#Bild wird mit der Webcam aufgenommen -> kein videostream, sondern ein "Foto" mit image = cap.read()
#src = cap.read()
image = cv.imread("rectanglepicture.png")
#Exception wenn "Image" leer ist -> prueft ob Bildquelle vorhanden ist
if image is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
cv.imshow('image', image)
#Konvertiert das Bild grau
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#Konvertiert das Bild in binaer
_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
for i, c in enumerate(contours):
    #Berechnet die Flaeche der Konturen
    area = cv.contourArea(c)
    #ignoriert zu große/kleine Konturen
    if area < 1e2 or 1e5 < area:
        continue
    #umrandet die Konturen in rot
    cv.drawContours(image, contours, i, (255, 0, 0), 2)
    #Findet die Ausrichtung jeder Form
    getOrientation(c, image)
cv.imshow('output', image)
cv.waitKey()
