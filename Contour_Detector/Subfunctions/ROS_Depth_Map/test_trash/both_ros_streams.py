#!/usr/bin/env python
import roslib
# roslib.load_manifest('kinect_pylistener')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np
import message_filters

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
    # self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 2, .1,allow_headerless=True)
    self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, .1)
    # self.ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
    self.ts.registerCallback(self.callback)
    print "test2"

  def callback(self,rgb_data, depth_data):
    try:
      image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
      depth_image = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
      depth_array = np.array(depth_image, dtype=np.float32)
      color_array = np.array(image, dtype=np.uint8)
      depth_array[np.isnan(depth_array)] = 0
      cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
      # print "depth_array"
      # print np.max(depth_array)
      # print np.min(depth_array)
      # print np.nanmax(depth_array)
      # print np.nanmin(depth_array)
      cv2.imshow('Depth Image', depth_array)
      cv2.imshow('RGB Image', image)
      cv2.waitKey(50) #too low results in lag
      # rospy.loginfo(image.shape)
      # cv2.imwrite('/home/andi/camera_rgb.jpg', color_array)
      # cv2.imwrite('/home/andi/camera_depth.pgm', depth_array*255)
      # print "test3"
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

if __name__ == '__main__':
    main(sys.argv)
