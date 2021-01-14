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


# while True:
def read_ros_depth_map(conn_depth):
	data_depth = b""
	payload_size = struct.calcsize("L")
	#receive depth map
	while len(data_depth) < payload_size:
		data_depth += conn_depth.recv(4096)
	packed_msg_size = data_depth[:payload_size]
	data_depth = data_depth[payload_size:]
	msg_size = struct.unpack("L", packed_msg_size)[0]
	while len(data_depth) < msg_size:
		data_depth += conn_depth.recv(4096)
	frame_data = data_depth[:msg_size]
	data_depth = data_depth[msg_size:]
	#load content from pickle
	frame_dict=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
	# extract depth map from transmitted dictionary
	dmap_raw=frame_dict[b"depthMap"]
	# depth map was encoded to 16-Bit PNG
	dmap = cv2.imdecode(dmap_raw.copy(), cv2.IMREAD_ANYDEPTH)
	#from dmap any pixel can be read like th e following:
	# print(dmap[int(height_depth/2),int(width_depth/2)]/1000, " meters")

	#display the peth image, there fore convert to 8-Bit and normalize
	cv_image_array = np.array(dmap.copy(), dtype = np.dtype('f8'))
	dmap_img_cv = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
	# height_depth = dmap_img_cv.shape[0]
	# width_depth = dmap_img_cv.shape[1]
	# #lets get the depth of the middle point!
	# print(dmap[int(height_depth/2),int(width_depth/2)]/1000, " meters")

	return dmap,dmap_img_cv

if __name__ == '__main__':
	# Parameters for receiving the image and button dictionary
	# print("hello")
	# os.system('python receive_ros_depth_map_stream.py')
	HOST_depth='localhost'
	PORT_depth=8090
	s_depth=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	print('ROS_DEPTH_STREAM Socket created')
	s_depth.bind((HOST_depth,PORT_depth))
	print('ROS_DEPTH_STREAM Socket bind complete')
	s_depth.listen(10)
	print('ROS_DEPTH_STREAM Socket now listening')
	conn_depth,addr=s_depth.accept()
	payload_size = struct.calcsize("L")

	print("lets go")
	while True:
		dmap,dmap_img_cv=read_ros_depth_map(conn_depth)
		cv2.imshow("depth_",dmap_img_cv)
		cv2.waitKey(1)
	cv2.destroyAllWindows()
	s_depth.shutdown(socket.SHUT_RDWR)
	s_depth.close()
