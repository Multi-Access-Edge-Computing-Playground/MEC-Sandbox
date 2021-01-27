import sys
#remove path to ROS path, so cv2 can work
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import imutils

alpha = 0.5
rgb = cv2.imread("RGB_Image.jpg")
dep = cv2.imread("Depth_Image.jpg",cv2.COLOR_BGR2GRAY)

h,w = rgb.shape[:2]
scale_dep = imutils.resize(dep, width=w)
cv2.imshow('rgb', rgb)
cv2.imshow('dep', scale_dep)
beta = (1.0 - alpha)
dst = cv2.addWeighted(rgb, alpha, scale_dep, beta, 0.0)
# [blend_images]
# [display]
cv2.imshow('dst', dst)
cv2.waitKey(0)
# [display]
cv2.destroyAllWindows()
