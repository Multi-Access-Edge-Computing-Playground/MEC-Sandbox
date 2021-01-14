import numpy as np
import cv2 as cv
import time

#from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
t0 = time.time()
img2 = cv.imread('scene_multi.jpg',0) # trainImage
# Initiate orb detector
orb = cv.SIFT_create()
t0 = time.time()
kp2, des2 = orb.detectAndCompute(img2,None)
print("it took ",round(time.time()-t0,5), " s to calculate Scene")
objs=['Obj1.jpg','Obj1.jpg','Obj3.jpg']
bounding_boxes=[]
for obj in objs:
    t0 = time.time()
    img1 = cv.imread(obj,0)          # queryImage
    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(img1,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        bounding_boxes.append(dst)
        # print("dst: ",dst)
        print("it took ",round(time.time()-t0,5), " s to find an Object")
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)
    #img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    #plt.imshow(img3, 'gray'),plt.show()
    #cv.imwrite('./feature.jpg',img3)
# draw bounding_boxes into the train image
for bbox in bounding_boxes:
    img2 = cv.polylines(img2,[np.int32(bbox)],True,255,3, cv.LINE_AA)
cv.imwrite('./feature_detect_multiple.jpg',img2)
