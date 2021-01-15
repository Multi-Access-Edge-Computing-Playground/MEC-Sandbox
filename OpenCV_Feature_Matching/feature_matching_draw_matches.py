import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./training.jpg',0)# queryImage
img2 = cv2.imread('./scene.jpg',0) # trainImage

# Initiate SIFT detector
print("1")
orb = cv2.ORB_create()
print("2")
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
print("3")
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
cv2.imshow("Detections", img3)
cv2.waitKey(0)
# plt.imshow(img3),plt.show()
print("thanks, i am done")
