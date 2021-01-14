#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import sys

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('./images/*.jpg')
[print(image) for image in images]
counter=0
for fname in images:
    print("processing image ...",fname)
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    # img = cv2.resize(img,(int(w*0.5), int(h*0.5)))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    temp_img = img.copy()
    temp_img = cv2.resize(temp_img,(int(1920), int(1080)))
    #cv2.imshow('img',temp_img)
    cv2.imwrite('img_%d.jpg'%counter,temp_img)
    counter+=1
    #cv2.waitKey(1)

#cv2.destroyAllWindows()
print("ran through all images, calibrating...")
h,w = img.shape[:2]

"""
Performing camera calibration by
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# store the camera matrix and distortion coefficients

import pickle
data = {"mtx": mtx, "dist": dist}
with open('calibration_data.pkl', 'wb') as handle:
    pickle.dump(data, handle)

#For testing lets use the matrix to undistort the chessboard image

with open('calibration_data.pkl', 'rb') as handle:
    data = pickle.load(handle)
mtx = data["mtx"]
dist = data["dist"]
#
# from tempfile import TemporaryFile
# outfile = TemporaryFile()
# np.savez(outfile, mtx, dist)
# # and load with every camera usage
# _ = outfile.seek(0) # Only needed here to simulate closing & reopening file
# sys.exit()
# npzfile = np.load(outfile)
# npzfile.files
# ['arr_0', 'arr_1']
# npzfile['arr_0']
# img = cv2.resize(img,(int(1920*0.75), int(1080*0.75)))

# img = cv.imread('left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# dst = cv2.undistort(img, data["mtx"], data["dist"]) # create undistorted image
cv2.imshow('corrected_img',dst)
cv2.waitKey(0)
cv2.imwrite('corrected_img.jpg',dst)
print(roi)
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
# # undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#
# # dst = cv2.undistort(img, data["mtx"], data["dist"]) # create undistorted image
# cv2.imshow('corrected_img',dst)
# cv2.waitKey(0)
# print(roi)
