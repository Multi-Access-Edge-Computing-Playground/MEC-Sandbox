import cv2
import io
import socket
import struct
import time
import pickle
#import zlib

HOST="192.168.1.11"#"raspberrypi"#'10.0.1.230'#'172.20.10.2'#'192.168.0.19'#'localhost'
PORT=8089

gui_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
gui_client_socket.connect((HOST, PORT))
connection = gui_client_socket.makefile('wb')

cam = cv2.VideoCapture(1)

cam.set(3, 320);
cam.set(4, 240);

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)


    print("{}: {}".format(img_counter, size))
    gui_client_socket.sendall(struct.pack("L", size) + data)
    img_counter += 1

cam.release()
