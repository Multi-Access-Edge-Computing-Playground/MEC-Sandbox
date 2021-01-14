#! /usr/bin/python
#Use python2 because ROS can only be installed with python2
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
import message_filters
HOST="localhost"
PORT=8090
ros_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ros_client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
not_connected=True
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
ros_count=0
while not_connected:
    try:
        ros_client_socket.connect((HOST, PORT))
        not_connected=False
    except:
        time.sleep(0.5)
        #loop until connected

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
    # self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 1)
    self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 4, .01,allow_headerless=True)
    self.ts.registerCallback(self.callback)
    print "test2"

  def callback(self,rgb_data, depth_data):
    try:
      global ros_count
      encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
      image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
      dmap = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
      #encode the depthmap because this way the datastream is much faster
      result, frame = cv2.imencode('.png', dmap.copy(),params=[cv2.CV_16U])
      result, rgb_frame = cv2.imencode('.jpg', image, encode_param)
      #to send depth map put frame in a dictionary
      dmap_dict={"depthMap":frame,"RGBimage":rgb_frame}
      #pickle
      data = pickle.dumps(dmap_dict, 0)
      size = len(data)
      #send
      ros_client_socket.sendall(struct.pack("L", size) + data)
      #we need to add a sleep timer since ros node needs to read the next image first
      # TODO use Multithreading to always have the latest image ready?
      # time.sleep(0.075)
      # time.sleep(0.2) #iPhone

      # ros_count+=1
      # print(ros_count)
    except CvBridgeError as e:
      print(e)



def main(args):
  print "test1"
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()
  ros_client_socket.shutdown(socket.SHUT_RDWR)
  ros_client_socket.close()

if __name__ == '__main__':
    main(sys.argv)
