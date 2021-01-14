#LSAT
"""
This script is used to detect objects with an orthogonal located camera,
pointing towards a 2D plane with objects on it.
Adapt parameters as required by the application e.g by changing the filter
for contour sizes --> if area < 100 or area > 100000
"""
# import the necessary packages
import contour_detector_picture as contour_detector
import time
import cv2
from datetime import datetime
import numpy as np

def undistort_image_with_calibration_data(calibration_data,image):
    # in main program get calibration_data from pickle file
    # import pickle
    # # fetch calibration data
    # with open('calibration_data.pkl', 'rb') as handle:
    #     calibration_data = pickle.load(handle)
    mtx = calibration_data["mtx"]
    dist = calibration_data["dist"]
    h,  w = image.shape[:2]
    # The size of the original chessborad calibration image must be the same for all other
    # images that shall be undistorted with that matrix!
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    # undistort
    img_undistorted = cv2.undistort(image, mtx, dist, None, newcameramtx)
    #now we can resize and work with the image
    #img1 = cv2.resize(img_undistorted,(int(w*0.5), int(h*0.5)))
    # display the image on screen and wait for a keypress
    h,  w = img_undistorted.shape[:2]
    # print("h, w: ",h," ",w,)
    return img_undistorted

def gray_blur_canny_thres_dilate_image(image,lower_thres=100,upper_thres=255):
    def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        # return the edged image
        return edged
    # imgGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # imgBlur = cv2.bilateralFilter(imgGray, 11, 17, 17)
    imgCanny = cv2.Canny(image,10,200)
    # imgCanny = auto_canny(imgBlur)
    _, imgThres = cv2.threshold(imgCanny, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # now add dilation to connect broken parts of an object
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    # imgThres = cv2.erode(imgThres, kernel, iterations=1)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)
    return imgThres

def find_draw_contours_calculate_px_pos_angle(imgThres,imgAnnotated):
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obj_middle_pts_list = []
    obj_angle_list = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        object_hierarchy = hierarchy[0][i][3] # -1 means has no parent
        if area < 100 or area > 100000 or object_hierarchy != -1:
            #skip contour
            continue
        cv2.drawContours(imgAnnotated, contours, i, (255, 0, 0), 2)
        # receive the angle and th emiddle pixel coordinate of the contour (object)
        angle, middle_px = contour_detector.getOrientation(c, imgAnnotated)
        obj_middle_pts_list.append(middle_px)
        obj_angle_list.append(angle)
    return imgAnnotated, obj_middle_pts_list, obj_angle_list

def main():
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, img = cap.read()
        # Our operations on the frame come here
        # img_undistorted = undistort_image_with_calibration_data(calibration_data,img)
        #crop the image to the area of interest
        # img = img_undistorted[547:1340,0:1644]
        imgAnnotated = img.copy() # this will be final image with drawings
        #prepare the image for detection
        imgThres = gray_blur_canny_thres_dilate_image(img,lower_thres=100)
        #find outer contours of objects
        imgAnnotated, obj_middle_pts_list, obj_angle_list = find_draw_contours_calculate_px_pos_angle(imgThres,imgAnnotated)
        # show the frame
        cv2.imshow("Detections", imgAnnotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
