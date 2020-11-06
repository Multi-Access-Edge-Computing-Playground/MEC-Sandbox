import cv2
import numpy as np
import sys
import time


if len(sys.argv)>1:
    inputImage = cv2.imread(sys.argv[1])
else:
    inputImage = cv2.imread("testQRCode.png")


# Display barcode and QR code location

def display(im, bbox):
    n = len(bbox[0])
    for j in range(n):
        cv2.line(im, tuple(bbox[0][j]), tuple(bbox[0][((j+1) % n)]), (255,255,0), 5)
    # Display results
    cv2.imshow("Results", im)

qrDecoder = cv2.QRCodeDetector()

# Detect and decode the qrcode
data,bbox,rectifiedImage = qrDecoder.detectAndDecode(inputImage)
if len(data)>0:
    print("Decoded Data : {}".format(data))
    display(inputImage, bbox)
    rectifiedImage = np.uint8(rectifiedImage);
    cv2.imshow("Rectified QRCode", rectifiedImage);
else:
    print("QR Code not detected")
    cv2.imshow("Results", inputImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
