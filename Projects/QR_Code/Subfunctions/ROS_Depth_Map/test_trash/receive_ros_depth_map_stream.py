#! /usr/bin/python
#TODO stream RGB and DEPTH simultaneously /
# Only works very laggy see "both_ros_streams.py"
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import socket
import struct
import pickle
import time
HOST="localhost"
PORT=8090
gui_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
not_connected=True
while not_connected:
    try:
        gui_client_socket.connect((HOST, PORT))
        not_connected=False
    except:
        time.sleep(0.5)
        #loop until connected

bridge = CvBridge()

def depth_image_callback(msg):
    try:
        #read depth map from ros publisher
        dmap = bridge.imgmsg_to_cv2(msg, "passthrough")
        #encode the depthmap because this way the datastream is much faster
        result, frame = cv2.imencode('.png', dmap.copy(),params=[cv2.CV_16U])
        #to send depth map put frame in a dictionary
        dmap_dict={"depthMap":frame}
        #pickle
        data = pickle.dumps(dmap_dict, 0)
        size = len(data)
        #send
        gui_client_socket.sendall(struct.pack("L", size) + data)

        #display the depth image
        cv_image_array = np.array(dmap.copy(), dtype = np.dtype('f8'))
        #normalize because 16bit images can not be displayed
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        height_depth = cv_image_norm.shape[0]
        width_depth = cv_image_norm.shape[1]
        vision_middle=(int(width_depth/2),int(height_depth/2))
        cv2.circle(cv_image_norm,vision_middle,4,(255,0,255),1)
        cv2.imshow("depth",cv_image_norm)
        cv2.waitKey(1)

    except CvBridgeError as e:
        print(e)

def rgb_image_callback(msg):
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("rgb",cv2_img)
        cv2.waitKey(1)

    except CvBridgeError as e:
        print(e)


def main():
    #stream settings and connect
    rospy.init_node('gui_client_socket', anonymous=True)
    # Define your image topic
    depth_image_topic = "/camera/depth/image_raw"
    rgb_image_topic = "/camera/color/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(depth_image_topic, Image, depth_image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
