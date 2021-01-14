#Detect Rectangle
from Subfunctions import functions_
from Subfunctions import highlevel_movements
import cv2
import imutils
import numpy as np
import sys
import os
import time
import threading
import socket

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
##Parameters
enable_stream=False
target_circle_size=20 #in pixels
max_speed=4 #degrees per second
min_speed=0.5 #degrees per second
action_time_interval=0.3
##
#enable streaming instead of local display
if enable_stream==True:
    #Variables to set
    HOST='192.168.79.15'#'192.168.0.10'
    PORT=8089
    #Lets create a socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    connection = client_socket.makefile('wb')
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

#Main
#Move Robot into position
# Import the utilities helper module
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Subfunctions import utilities
# Parse arguments
args = utilities.parseConnectionArguments()
# Create connection to the device and get the router
with utilities.DeviceConnection.createTcpConnection(args) as router:
    # Create required services
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)
    # Example core
    success = True
    success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.02,-0.08,0.78,90,3,3])
    #Start
    cap = cv2.VideoCapture('rtsp://192.168.1.10/color')
    t0=time.time()
    while(True):
        ret, frame = cap.read()
        #crop the frame so the grippertip is gone
        (h, w) = frame.shape[:2]
        frame=frame[0:int(h*0.88)]
        #Place the middle circle
        (h, w) = frame.shape[:2]
        vision_middle=(int(w/2),int(h/2))
        cv2.circle(frame,vision_middle,target_circle_size*2,(255,0,255),1)
        #get the target circle
        image,target_middle,success_flag=functions_.detect_rectangle_middle(frame)

        #cv2.imshow('frame',image)
        if enable_stream==True:
            #Send the image to the client
            result, frame = cv2.imencode('.jpg', image, encode_param)
            #data = zlib.compress(pickle.dumps(frame, 0))
            data = pickle.dumps(frame, 0)
            size = len(data)
            client_socket.sendall(struct.pack("L", size) + data)
        else:
            cv2.imshow('frame',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 base.Stop()
                 break

        if success_flag==False:
            #contour was not detected
            target_middle=vision_middle
        elif time.time()-t0>action_time_interval:
            #calculate the pose increment to hit the target
            functions_.rotate_to_target(target_middle,vision_middle,w,h,max_speed,min_speed,
                target_circle_size,base)
            t0=time.time()


cap.release()
cv2.destroyAllWindows()
