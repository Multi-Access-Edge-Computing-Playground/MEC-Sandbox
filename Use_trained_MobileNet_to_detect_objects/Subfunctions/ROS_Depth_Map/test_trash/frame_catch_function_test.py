import receive_from_socket_stream
import sys
#remove path to ROS path, so cv2 can work
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import socket
import struct
import pickle
import cv2
import numpy as np
import base64
import os

# read_ros_depth_map

# Parameters for receiving the image and button dictionary
# print("hello")
# os.system('python receive_ros_depth_map_stream.py')

## Copy this part into Script
HOST_depth='localhost'
PORT_depth=8090
s_depth=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('ROS_DEPTH_STREAM Socket created')
s_depth.bind((HOST_depth,PORT_depth))
print('ROS_DEPTH_STREAM Socket bind complete')
s_depth.listen(10)
print('ROS_DEPTH_STREAM Socket now listening')
conn_depth,addr=s_depth.accept()


print("lets go")
while True:
    #in while loop call this function
    dmap,dmap_img_cv=receive_from_socket_stream.read_ros_depth_map(conn_depth)

    cv2.imshow("depth_",dmap_img_cv)
    cv2.waitKey(1)
cv2.destroyAllWindows()
s_depth.shutdown(socket.SHUT_RDWR)
s_depth.close()
