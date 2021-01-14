# in based on button_cv_test2.py
import numpy as np
import cv2
import math
import imutils
# test_image_names=["IMG_0568.JPEG","IMG_0573.JPEG","IMG_0575.JPEG","IMG_0581.JPEG","IMG_0592.JPEG","IMG_0606.JPEG"]
padding = 0
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def find_buttons_in_bbox(frame,bounding_boxes,whole_frame=False):
	#if whole_frame == True, contours for the whole image will be found
	button_bbox_all=[] #will receive all bounding boxes (those must not be rectangles)
	button_bbox=[]
	if whole_frame == False:
		for bounding_box in bounding_boxes:
			# print(fbox)
			bbox = bounding_box.bounding_box.flatten().tolist()
			bbox = list(int(x) for x in bbox)
			# bbox=list(bounding_box.bbox)
			x0, y0, x1, y1 = bbox
			bbox_img = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
			if bbox_img.size ==0:
				continue
			# img = cv2.imread("./test_images/"+image_name)
			# img = imutils.resize(img, height=1080)
			# Prepocess
			gray = cv2.cvtColor(bbox_img,cv2.COLOR_BGR2GRAY)
			blur = cv2.bilateralFilter(gray, 11, 17, 17)
			# flag, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
			# #Canny Edge detector
			edged = auto_canny(blur)
			# Find contours
			contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			contours = sorted(contours, key=cv2.contourArea,reverse=True)

			# # get all the perimeters
			# perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
			# # Select long perimeters only
			# filtered_cnts_1=[contours[i] for i in range(len(perimeters)) if 200<perimeters[i]<2000] #perimeter in pixel
			# #add function that checks if contour is a box
			# filtered_cnts_3=[]
			# for i in range(len(filtered_cnts_1)):
			# 	hull = cv2.convexHull(filtered_cnts_1[i])
			# 	hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
			# 	if 4<=len(hull)<10:
			# 		filtered_cnts_3.append(hull)

			# Show image
			# imgcont = img.copy()
			# for cnt in filtered_cnts_3:
			for cnt in contours:
				cnt[:, 0][:,0] += x0
				cnt[:, 0][:,1] += y0
				# pose_translation=list(pose_mat[:-1,3])
				cv2.drawContours(frame, [cnt], 0, (0,255,0), 1)
				#convert contours to boundingboxes
				button_bbox=[]
				for pt in cnt[:,:][:,0][:,:]:
					button_bbox.append(list(pt))
				# print(button_bbox)
			button_bbox_all.append(button_bbox) #has all the bounding boxes
			# [cv2.drawContours(frame, [cnt], 0, (0,255,0), 2) for cnt in filtered_cnts_3]
	else:
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		blur = cv2.bilateralFilter(gray, 11, 17, 17)
		# flag, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
		# #Canny Edge detector
		edged = auto_canny(blur)
		# Find contours
		contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea,reverse=True)
		for cnt in contours:
			# pose_translation=list(pose_mat[:-1,3])
			cv2.drawContours(frame, [cnt], 0, (0,255,0), 1)
			#convert contours to boundingboxes
			button_bbox=[]
			for pt in cnt[:,:][:,0][:,:]:
				button_bbox.append(list(pt))
			# print(button_bbox)
		button_bbox_all.append(button_bbox) #has all the bounding boxes
	return frame, button_bbox_all
