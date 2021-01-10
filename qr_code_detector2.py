import cv2
import numpy as np
import sys
import time


if len(sys.argv)>1:
    inputImage = cv2.imread(sys.argv[1])
else:
<<<<<<< HEAD
    inputImage = cv2.imread("testQRCode.png")
=======
    inputImage = cv2.imread("testQRCode2.png")
>>>>>>> 30a7a658c1a4f7fb28552f7c1eabea3fef9fb3b6


# Display barcode and QR code location

def display(im, bbox):
<<<<<<< HEAD
    n = len(bbox[0])
    for j in range(n):
        cv2.line(im, tuple(bbox[0][j]), tuple(bbox[0][((j+1) % n)]), (255,255,0), 5)
=======
    print("bbox as np.array:\n",bbox)
    #lets convert numpy-array to list (because numpy arrays suck.)
    bbox = bbox.tolist()[0]
    #then convert values to integer, since those are pixel coordinates
    new_bbox = []
    for pixel_pair in bbox:
        pair=[]
        for pixel_coordinate in pixel_pair:
            pair.append(int(pixel_coordinate))
        new_bbox.append(pair)
    print("bbox as List:",new_bbox)
    n = len(new_bbox)

    for j in range(n):
        cv2.line(im, tuple(new_bbox[j]), tuple(new_bbox[ (j+1) % n]), (255,0,0), 3)
>>>>>>> 30a7a658c1a4f7fb28552f7c1eabea3fef9fb3b6
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
