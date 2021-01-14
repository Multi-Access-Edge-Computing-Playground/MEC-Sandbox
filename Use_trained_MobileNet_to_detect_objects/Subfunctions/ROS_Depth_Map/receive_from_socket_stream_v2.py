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
from threading import Thread
import time

class Receive_Frame_And_Dmap:
	#call with
	# HOST='localhost'
	# PORT=8090
	# vs = Receive_Frame_And_Dmap().start(HOST,PORT)
	# success, dmap, dmap_img_cv, rgb = vs.read()
	# stop with
	# vs.stop()
	def __init__(self):
		self.dmap = None
		self.rgb = None
		self.dmap_img_cv = None
		self.success = False
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		# initialize the socket
		self.conn=""

	def start(self,HOST,PORT):
		# start the thread to read frames from the video stream
		# therefore bind to host
		s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		print('ROS_DEPTH_STREAM Socket created')
		s.bind((HOST,PORT))
		print('ROS_DEPTH_STREAM Socket bind complete')
		s.listen(10)
		print('ROS_DEPTH_STREAM Socket now listening')
		self.conn, address = s.accept()
		ip, port = str(address[0]), str(address[1])
		print("Connected with " + ip + ":" + port)
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		data = b""
		payload_size = struct.calcsize("L")
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				self.conn.close()
				return
			# otherwise, read the next frame and button_dict from the stream
			#here we listen on port
			#receive image and button dict
			while len(data) < payload_size:
				data += self.conn.recv(2048*2)
			packed_msg_size = data[:payload_size]
			data = data[payload_size:]
			msg_size = struct.unpack("L", packed_msg_size)[0]
			while len(data) < msg_size:
				data += self.conn.recv(2048*2)
			frame_data = data[:msg_size]
			data = data[msg_size:]
			frame_dict=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
			# extract depth map from transmitted dictionary
			dmap_raw=frame_dict[b"depthMap"]
			rgb_raw=frame_dict[b"RGBimage"]
			# depth map was encoded to 16-Bit PNG
			self.dmap = cv2.imdecode(dmap_raw.copy(), cv2.IMREAD_ANYDEPTH)
			self.rgb = cv2.imdecode(rgb_raw.copy(), cv2.IMREAD_COLOR)
			#from dmap any pixel can be read like th e following:
			# print(dmap[int(height_depth/2),int(width_depth/2)]/1000, " meters")

			#display the peth image, there fore convert to 8-Bit and normalize
			cv_image_array = np.array(self.dmap.copy(), dtype = np.dtype('f8'))
			self.dmap_img_cv = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
			# height_depth = dmap_img_cv.shape[0]
			# width_depth = dmap_img_cv.shape[1]
			# #lets get the depth of the middle point!
			# print(dmap[int(height_depth/2),int(width_depth/2)]/1000, " meters")
			self.success = True

	def read(self):
		# return the frame most recently read
		return self.success, self.dmap, self.dmap_img_cv, self.rgb

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

# In development
class Send_Frame_And_Dict:
	#call with
	# HOST='localhost', "172.20.10.2", "192.168.168.125"
	# PORT=8089
	# sender_thread = Send_Frame_And_Dict().start(HOST,PORT)
	# sender_thread.send(frame_dict)
	# stop with
	# sender_thread.stop()
	def __init__(self):
		# self.imageFile = None
		# self.buttonDict = None
		self.frame_dict = None
		self.success = False
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		# initialize the socket
		self.gui_client_socket=""
		self.token=0
		# self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

	def start(self,HOST="localhost",PORT=8089):
		#stream settings for GUI stream
		self.gui_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.gui_client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.gui_client_socket.connect((HOST, PORT))
		connection = self.gui_client_socket.makefile('wb')
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				self.gui_client_socket.close()
				return
			if self.frame_dict is not None and self.token>0:
				data = pickle.dumps(self.frame_dict,fix_imports=True)
				size = len(data)
				self.gui_client_socket.sendall(struct.pack("L", size) + data)
				self.success = True
				self.token-=0
				if self.token==-1:
					self.token=0


	def send(self,frame_dict):
		self.frame_dict = frame_dict
		self.token+=1 #updater is allowed to send

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

# while True:
# def read_ros_depth_map(conn_depth):
#	 data_depth = b""
#	 payload_size = struct.calcsize("L")
#	 #receive depth map
#	 while len(data_depth) < payload_size:
#		 data_depth += conn_depth.recv(4096)
#	 packed_msg_size = data_depth[:payload_size]
#	 data_depth = data_depth[payload_size:]
#	 msg_size = struct.unpack("L", packed_msg_size)[0]
#	 while len(data_depth) < msg_size:
#		 data_depth += conn_depth.recv(4096)
#	 frame_data = data_depth[:msg_size]
#	 data_depth = data_depth[msg_size:]
#	 #load content from pickle
#	 frame_dict=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
#	 # extract depth map from transmitted dictionary
#	 dmap_raw=frame_dict[b"depthMap"]
#	 rgb_raw=frame_dict[b"RGBimage"]
#	 # depth map was encoded to 16-Bit PNG
#	 dmap = cv2.imdecode(dmap_raw.copy(), cv2.IMREAD_ANYDEPTH)
#	 rgb = cv2.imdecode(rgb_raw.copy(), cv2.IMREAD_COLOR)
#	 #from dmap any pixel can be read like th e following:
#	 # print(dmap[int(height_depth/2),int(width_depth/2)]/1000, " meters")
#
#	 #display the peth image, there fore convert to 8-Bit and normalize
#	 cv_image_array = np.array(dmap.copy(), dtype = np.dtype('f8'))
#	 dmap_img_cv = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
#	 # height_depth = dmap_img_cv.shape[0]
#	 # width_depth = dmap_img_cv.shape[1]
#	 # #lets get the depth of the middle point!
#	 # print(dmap[int(height_depth/2),int(width_depth/2)]/1000, " meters")
#
#	 return dmap,dmap_img_cv,rgb

if __name__ == '__main__':
	#Testing
	# Parameters for receiving the image and button dictionary
	# print("hello")
	# os.system('python receive_ros_depth_map_stream.py')
	HOST_depth='localhost'
	PORT_depth=8090
	vs = Receive_Frame_And_Dmap().start(HOST_depth,PORT_depth)
	#init framerate display
	t0=time.time()
	frame_count=0
	framerate=0
	fontStyle = cv2.FONT_HERSHEY_SIMPLEX
	t_delay=time.time()
	print("lets go")
	while True:
		success, dmap, dmap_img_cv, rgb = vs.read()
		if success==True:
			cv2.putText(rgb, "Framerate: "+str(framerate), (10,15),fontStyle, 0.5, (0, 0, 255), 2)
			cv2.imshow("depth_",dmap_img_cv)
			cv2.imshow("rgb",rgb)
			key = cv2.waitKey(1)
			if (key == 27) or (key == 13):
				break
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			# cv2.waitKey(1)
			t1=time.time()
			frame_count+=1
			if t1-t0>=1.0:
				framerate=frame_count
				frame_count=0
				t0=time.time()
		while time.time()-t_delay<0.03:
			time.sleep(0.003)
		t_delay=time.time()
	cv2.destroyAllWindows()
	vs.stop()
