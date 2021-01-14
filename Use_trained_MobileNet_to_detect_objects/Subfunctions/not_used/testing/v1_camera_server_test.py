import numpy as np
import cv2
import socket
import struct
import pickle

cap = cv2.VideoCapture(2)

#stream settings for GUI stream
localhost=True
if localhost==True:
    HOST="localhost"#"172.20.10.2"#"raspberrypi"#'10.0.1.230'#'172.20.10.2'#'192.168.0.19'#'localhost'
else:
    HOST="172.20.10.2"
    # HOST="192.168.168.125"
PORT=8089
gui_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
gui_client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

gui_client_socket.connect((HOST, PORT))
connection = gui_client_socket.makefile('wb')
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
temporary_dict_copy={1:"hello world", 2:"hello mars"}

while(True):
    # Capture frame-by-frame
    ret, rgb_frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(rgb_frame.copy(), cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    result, rgb_frame = cv2.imencode('.jpg', rgb_frame, encode_param)
    frame_dict={"imageFile":rgb_frame,"buttonDict":temporary_dict_copy}
    data = pickle.dumps(frame_dict,fix_imports=True)
    #alternate pickle version

    size = len(data)
    gui_client_socket.sendall(struct.pack(">L", size) + data)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
