
import cv2
import numpy as np
import time
import sys
from scipy.spatial.transform import Rotation as R

cap = cv2.VideoCapture(0)
qrDecoder = cv2.QRCodeDetector()


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    #Barcode decoded and detected
    decodedBarcode, bbox, _ = qrDecoder.detectAndDecode(frame)


    #Box around the Barcode
    if bbox is not None:
        n = len(bbox[0])
        for j in range(n):
            cv2.line(frame, tuple(bbox[0][j]), tuple(bbox[0][((j+1) % n)]), (255,255,0), 5)
        #prints out the decoded Barcode
        print(decodedBarcode)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
