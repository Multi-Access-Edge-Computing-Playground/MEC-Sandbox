import cv2
import numpy as np
from imutils.video import WebcamVideoStream
import time
import sys
from scipy.spatial.transform import Rotation as R
# cap = WebcamVideoStream(0).start()
# time.sleep(0.5)
frame_count=0
framerate=0
qrCodeDetector = cv2.QRCodeDetector()
fontStyle = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
fontThickness = 2
t0=time.time()
while True:
    # image = cap.read()
    # image = cv2.imread('testQRCode2.png')
    image = cv2.imread('testQRCode.png')
    decodedText, points, _ = qrCodeDetector.detectAndDecode(image)
    if points is not None:
        # print(points)
        nrOfPoints = len(points[0])
        for i0 in range(nrOfPoints):
            nextPointIndex = (i0+1) % nrOfPoints
            cv2.line(image, tuple(points[0][i0]), tuple(points[0][nextPointIndex]), (255,0,0), 5)
        # print(decodedText)
        # rect = order_points(pts)
        (tl, tr, br, bl) = points[0]
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(points[0], dst)
        # https://stackoverflow.com/questions/60138100/calculate-pitch-and-roll-and-yaw-by-4-points-detected-from-a-square-in-opencv
        print(M)
        constellations = ["xyz","xzy","yxz","yzx","zxy","zyx","XYZ","XZY","YXZ","YZX","ZXY","ZYX"]
        for const in constellations:
            euler = R.from_matrix(M).as_euler(const,degrees=True)
            # print(round(euler,2))
            print([round(elem,2) for elem in euler])
        sys.exit()
    t1=time.time()
    frame_count+=1
    if t1-t0>=1.0:
        framerate=frame_count
        frame_count=0
        t0=time.time()
    cv2.putText(image, "Framerate: "+str(framerate)+" Framecount: "+str(frame_count),
                (10,15),fontStyle, 0.5, (255,255,255), 2)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
