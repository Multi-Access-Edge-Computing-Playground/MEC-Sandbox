#! /usr/bin/python
#TODO stream RGB and DEPTH simultaneously
"""
To start two terminals must be executed with the following commands
roscore
roslaunch kinova_vision kinova_vision.launch
"""
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import sys
# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import io
import socket
import struct
import pickle
import base64
# HOST="localhost"#"raspberrypi"#'10.0.1.230'#'172.20.10.2'#'192.168.0.19'#'localhost'
# PORT=8089
# gui_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# gui_client_socket.connect((HOST, PORT))
# encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
# # global gui_client_socket
# # Instantiate CvBridge
# bridge = CvBridge()

# while True:
# src = "./one_depth_map.pkl"
# data = open(src).read().replace('\r\n', '\n') # read and replace file contents
# dst = src + ".tmp"
# open(dst, "w").write(data) # save a temporary file

# dmap = pickle.load(open(dst, "r"))
dmap0 = pickle.load(open("one_depth_map.pkl", "r" ) )
pickle.dump(dmap0, open('filename_out', 'wb'))
dmap = pickle.load(open("filename_out", "rb" ) )
print(type(dmap))

sys.exit()

def depth_image_callback(msg):
    # print("Received an image!")
    # global gui_client_socket
    try:
        dmap = bridge.imgmsg_to_cv2(msg, "32FC1")
        #Only Temporary to test offline
        with open("one_depth_map.pkl", "wb") as f:
            pickle.dump(dmap, f)
        ###
        cv_image_array = np.array(dmap, dtype = np.dtype('f8'))
        print(cv_image_array)
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        height_depth = cv_image_norm.shape[0]
        width_depth = cv_image_norm.shape[1]
        #lets get the depth of the middle point!
        print(dmap[int(height_depth/2),int(width_depth/2)]/1000, " meters")
        vision_middle=(int(width_depth/2),int(height_depth/2))
        cv2.circle(cv_image_norm,vision_middle,4,(255,0,255),1)
        cv2.imshow("depth",cv_image_norm)
        cv2.waitKey(1)
        #Option A:
        # s = base64.b64encode(cv_image_array)
        # Option B:
        # result, frame = cv2.imencode('.jpg', dmap, encode_param)
        # dmap_dict={"depthMap":s}
        #Option C:
        dmap_dict={"depthMap":cv_image_array}
        # dmap_dict={"depthMap":frame}
        # dmap_dict={"depthMap":dmap}
        data = pickle.dumps(dmap_dict, 0)
        size = len(data)
        gui_client_socket.sendall(struct.pack("L", size) + data)

    except CvBridgeError as e:
        print(e)

def rgb_image_callback(msg):
    # print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("rgb",cv2_img)
        cv2.waitKey(1)

    except CvBridgeError as e:
        print(e)


# def main():
#     #stream settings and connect
#
#     # gui_client_socket.listen(10)
#     # conn,addr = gui_client_socket.accept()
#     #
#     # connection = gui_client_socket.makefile('wb')
#     rospy.init_node('gui_client_socket', anonymous=True)
#     # Define your image topic
#     # image_topic = "/cameras/left_hand_camera/image"
#     depth_image_topic = "/camera/depth/image_raw"
#     rgb_image_topic = "/camera/color/image_raw"
#     # image:=/camera/depth/image_raw
#     # Set up your subscriber and define its callback
#     rospy.Subscriber(depth_image_topic, Image, depth_image_callback)
#     # rospy.Subscriber(rgb_image_topic, Image, rgb_image_callback)
#     # Spin until ctrl + c
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#       print("Shutting down")
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()
