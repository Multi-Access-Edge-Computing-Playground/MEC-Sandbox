import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import math
import imutils
import time
import sys
test_image_names=["IMG_0568.JPEG","IMG_0573.JPEG","IMG_0575.JPEG","IMG_0581.JPEG","IMG_0592.JPEG","IMG_0606.JPEG"]
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged
for image_name in test_image_names:

    img = cv2.imread("./test_images/"+image_name)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t0=time.time()
    img = imutils.resize(img, height=1080)
    # img = imutils.resize(img, width=300)
    # plt.imshow(img)
    # cv2.imshow("img", img)

    # Prepocess
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    # blur = cv2.bilateralFilter(gray, 5, 7, 7)

    # blur = cv2.GaussianBlur(gray,(3,3),1000)
    # blur = gray
    # cv2.imshow("blur", blur)


    flag, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)

    # edged = cv2.Canny(blur, 30, 200)
    edged = auto_canny(blur)
    # edged = cv2.Canny(blur, 10, 200)
    # cv2.imshow("edged", edged)
    # Find contours
    # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)
    print("number of contours: ",len(contours))
    # get all the perimeters
    perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
    #print("the perimeters: ",perimeters)
    # Select long perimeters only
    filtered_cnts_1=[contours[i] for i in range(len(perimeters)) if 200<perimeters[i]<2000] #perimeter in pixel
    # listindex=[i for i in range(len(perimeters)) if 100<perimeters[i]<3000] #perimeter in pixel
    # numcards=len(listindex)

    # #Canny Edge detector
    # card_number = -1 #just so happened that this is the worst case
    # stencil = np.zeros(img.shape).astype(img.dtype)
    # cv2.drawContours(stencil, [filtered_cnts_1[card_number]], 0, (255, 255, 255), cv2.FILLED)
    # res = cv2.bitwise_and(img, stencil)
    # # cv2.imwrite("out", res)
    # cv2.imshow("out", res)
    # canny = cv2.Canny(res, 100, 200)
    # # cv2.imwrite("canny.bmp", canny)
    # cv2.imshow("canny", canny)
    #continue here https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    #add function that checks if contour is a box
    # filtered_cnts_2=[]
    # for i in range(len(filtered_cnts_1)):
    #     epsilon = 0.05*cv2.arcLength(filtered_cnts_1[i],True)
    #     approx = cv2.approxPolyDP(filtered_cnts_1[i],epsilon,True)
    #     if 4<=len(approx)<5:
    #         filtered_cnts_2.append(approx)
    filtered_cnts_3=[]
    for i in range(len(filtered_cnts_1)):
        hull = cv2.convexHull(filtered_cnts_1[i])
        hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
        if 4<=len(hull)<10:
            filtered_cnts_3.append(hull)

    # Show image
    imgcont = img.copy()
    # [cv2.drawContours(imgcont, [contours[i]], 0, (0,255,0), 2) for i in listindex]
    x0=10
    y0=20
    for cnt in filtered_cnts_3:
        print(cnt)
        cnt[:, 0][:,0] += x0
        cnt[:, 0][:,1] += y0
        print(cnt)
        print(list(cnt[:,:][:,0][:,:]))
        button_bbox=[]
        for pt in cnt[:,:][:,0][:,:]:
            button_bbox.append(list(pt))
        print(button_bbox)
        sys.exit()
        # pose_translation=list(pose_mat[:-1,3])
        cv2.drawContours(imgcont, [cnt], 0, (0,255,0), 2)
    # [cv2.drawContours(imgcont, [cnt], 0, (0,255,0), 2) for cnt in filtered_cnts_3]
    print("took ",time.time()-t0," seconds")
    # plt.imshow(imgcont)
    cv2.imshow("imgcont", imgcont)
    cv2.waitKey(0)
