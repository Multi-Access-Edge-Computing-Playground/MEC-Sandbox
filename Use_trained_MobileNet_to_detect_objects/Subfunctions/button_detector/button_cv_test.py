# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2


# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
test_image_names=["IMG_0568.JPEG","IMG_0573.JPEG","IMG_0575.JPEG","IMG_0581.JPEG","IMG_0592.JPEG","IMG_0606.JPEG"]
for image_name in test_image_names:
    image = cv2.imread("./test_images/"+image_name)
    resized = imutils.resize(image, width=300)
    # resized = imutils.resize(image, height=1080)

    ratio = image.shape[0] / float(resized.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Thresh", thresh)
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()
    boxes=[]
    # loop over the contours
    for c in cnts:
    	# compute the center of the contour, then detect the name of the
    	# shape using only the contour
    	M = cv2.moments(c)
    	shape, approx = sd.detect(c)
    	if M["m00"] != 0:
        	cX = int((M["m10"] / M["m00"]) * ratio)
        	cY = int((M["m01"] / M["m00"]) * ratio)
        	# multiply the contour (x, y)-coordinates by the resize ratio,
        	# then draw the contours and the name of the shape on the image
        	c = c.astype("float")
        	c *= ratio
        	c = c.astype("int")
        	approx = approx.astype("float")
        	approx *= ratio
        	approx = approx.astype("int")
        	# filter by size and name
        	if 10<M['m00']<5000:# and shape=="box":
        		print(shape," ",M['m00'])
        		# cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        		cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)
        		cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
        		boxes.append(approx)
        #Use the boxes to craete a 4x3 box matrix and approximate missing boxes
        #TODO boxes...
        	# show the output image
    image = imutils.resize(image, height=1080)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
