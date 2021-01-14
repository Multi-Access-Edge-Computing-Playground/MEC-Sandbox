#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import sys
import pickle

# Extracting path of individual image stored in a given directory
images = glob.glob('./images/*.jpg')
[print(image) for image in images]
counter=0
with open('calibration_data.pkl', 'rb') as handle:
    data = pickle.load(handle)
mtx = data["mtx"]
dist = data["dist"]



for fname in images:
    print("processing image ...",fname)
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    # The size of the original chessborad calibration image must remain for all other
    # images that shall be undistorted
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #now we can resize and work with the image
    img1 = cv2.resize(dst.copy(),(int(w*0.25), int(h*0.25)))

    # dst = cv2.undistort(img, data["mtx"], data["dist"]) # create undistorted image
    cv2.imshow('corrected_img',img1)
    cv2.waitKey(0)

cv2.destroyAllWindows()
