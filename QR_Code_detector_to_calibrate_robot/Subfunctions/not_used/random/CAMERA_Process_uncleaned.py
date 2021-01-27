#!/usr/bin/env python3
"""
The States:
CAM_ID = [1,0] detect objects + click on box
CAM_ID = [2,0] detect objects
CAM_ID = [3,0] detect objects + show options (follow, inspect)
CAM_ID = [3,1] detect objects + (msg: moving robot to orthogonal position)
CAM_ID = [3,2] detect objects + (msg: move dog closer)
CAM_ID = [4,0] detect buttons + clickable buttons + option to move back and forth
CAM_ID = [4,1] detect buttons + clickable buttons + msg: moving away
CAM_ID = [4,2] detect buttons + clickable buttons + msg: moving closer
CAM_ID = [4,3] msg: tapping button

"""
import sys
import math
import json
#remove path to ROS path, so cv2 can work
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# from imutils.video import WebcamVideoStream
from threading import Thread
from queue import Queue
import imutils
import time
from datetime import datetime
import os
from PIL import Image
import cv2
from Subfunctions import detect
from Subfunctions import functions_
from Subfunctions import orientation_regulator_final
from Subfunctions.button_detector import button_cv_final
# import tflite_runtime.interpreter as tflite
import platform
from operator import add,sub
import multiprocessing

import numpy as np
# import screeninfo
import time
from scipy.spatial.transform import Rotation as R
import math
#stream libraries
import io
import socket
import struct
import pickle
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
record_video=True
camera="Kinova"
# camera="Orbbec"

if camera=="Kinova":
	from Subfunctions.ROS_Depth_Map import receive_from_socket_stream_v2
else:
	from openni import openni2#, nite2
	from openni import _openni2 as c_api

class Depth_Map_Params():
	def __init__(self,ratio_width=0,ratio_height=0,width_dep=0,height_dep=0,fov_width=0,fov_height=0,distance_obj=0.0):
		self.ratio_width = ratio_width
		self.ratio_height = ratio_height
		self.width_dep = width_dep
		self.height_dep = height_dep
		self.width_rgb = "TBD"
		self.height_rgb = "TBD"
		self.fov_width = fov_width
		self.fov_height = fov_height
		self.distance_obj = 0 # changes with every call of child functions
		self.robot_range_limit = 0.9
		self.alignment = False # True when rgb and depth image have to be aligned
		#the following parameters are only used when rgb and depth are not aligned
		self.dmatfri = [] #y,x depth_map_area_to_fit_rgb_image
		self.alignment_fov_angle_x_above = 0 #degrees
		self.alignment_fov_angle_x_below = 0 #degrees
		self.alignment_fov_angle_y_above = 0 #degrees
		self.alignment_fov_angle_y_below = 0 #degrees
		self.alignment_fov_angles = [] # [Rx,Ry], degrees must be set for kinova and other cameras
		self.alignment_offset_y = 0
		self.alignment_offset_x = 0
		self.alignment_null_coordinate = [] # [Px,Py], pixels must be set for kinova and other cameras


	def px_point2xyz(self,px_point,dmap,current_joint_position):
		point_depth=[int(px_point[0]/float(self.ratio_width)),int(px_point[1]/float(self.ratio_height))]
		self.distance_obj=dmap[point_depth[1],point_depth[0]]/1000 #meters
		pose_in_vectorspace=[0,0,0]
		if self.distance_obj!=0.0:
			if self.alignment == False:
				px=point_depth[0]-self.width_dep/2
				py=point_depth[1]-self.height_dep/2
				degrees_x=px*(self.fov_width/2)/(self.width_dep/2)
				degrees_y=py*(self.fov_height/2)/(self.height_dep/2)
				cam_shift=[0,0,self.distance_obj,degrees_y,-degrees_x,0]
			elif self.alignment == True:
				# since RGB and Depth stream are not aligned in Kinova
				# the field of view and null coordinate from dmap must be adapted
				# in order to calculate the correct angles
				px=point_depth[0]-self.alignment_null_coordinate[0]
				py=point_depth[1]-self.alignment_null_coordinate[1]

				#Koordinatenursprung liegt bei Kinova nicht in der Mitte des Frames
				#degreees y
				if point_depth[1]<self.alignment_null_coordinate[1]: #below
					degrees_y = py*self.alignment_fov_angle_y_below/(
						self.alignment_null_coordinate[1])
				elif point_depth[1]>=self.alignment_null_coordinate[1]: #above
					degrees_y = py*self.alignment_fov_angle_y_above/(
						self.height_dep-self.alignment_null_coordinate[1])
				#degrees x
				if point_depth[0]<self.alignment_null_coordinate[0]: #below
					degrees_x = px*self.alignment_fov_angle_x_below/(
						self.alignment_null_coordinate[0])
				elif point_depth[0]>=self.alignment_null_coordinate[0]: #above
					degrees_x = px*self.alignment_fov_angle_x_above/(
						self.width_dep-self.alignment_null_coordinate[0])

				cam_shift=[0,0,self.distance_obj,degrees_y,-degrees_x,0]
			# print(degrees_x," / ",degrees_y)
			#Calculate the vectorspace pose
			# pose_mat = functions_.forward_kin(current_joint_position) #get 4x4 matrix from joint angles

			temp_mat, pose_in_vectorspace = functions_.forward_kin_plus(current_joint_position,cam_shift) #get 4x4 matrix from joint angles
			# rot_x_iteration, temp_pose = posetrans_mat(pose_mat,[0,0,0,0,-degrees_x,0])
			# rot_y_iteration, temp_pose = posetrans_mat(rot_x_iteration,[0,0,0,degrees_y,0,0])
			# #test
			# rot_y_iteration, temp_pose = posetrans_mat(pose_mat,[0,0,0,degrees_y,-degrees_x,0])
			# temp_mat, pose_in_vectorspace = posetrans_mat(rot_y_iteration,[0,0,self.distance_obj,0,0,0])
			pose_in_vectorspace=[round(num, 3) for num in pose_in_vectorspace]
		else:
			#raise Exception("Depth was not read correctly (0)")
			None
		return pose_in_vectorspace

	def bbox2plane(self,rectangle,dmap,current_joint_position):
		x0, y0, x1, y1 = rectangle
		#TODO make sure rectangle is cropped to image boundary
		# if x0>self.width_rgb:
		# 	x0=self.width_rgb
		# 	# soemthing like https://answers.opencv.org/question/70953/roi-out-of-bounds-issue/
		pts=[[x0,y0],[x1,y0],[x0,y1],[x1,y1]]
		rectangle_xyz=[]
		for pt in pts:
			pose_in_vectorspace=self.px_point2xyz(pt,dmap,current_joint_position)[0:3]
			rectangle_xyz.append(pose_in_vectorspace)
		return rectangle_xyz #the 4 edges with xyz coordinates in a list
	def get_pt_on_orthogonal_vector(self,rectangle,dmap,current_joint_position,distance):
		points = self.bbox2plane(rectangle,dmap,current_joint_position)
		#check if enough points have depth coordinates (sometimes depth is 0)
		correct_points=[]
		for pt in points:
			if pt!=[0,0,0]:
				correct_points.append(pt)
		correct_points_list=correct_points.copy()
		if len(correct_points)==3:
			D1,D3,D4 = np.array(correct_points)
		elif len(correct_points)==4:
			D1,D2,D3,D4 = np.array(correct_points)
		else:
			# raise Exception("not enough valid points to calculate plane")
			return [0,0,0], correct_points_list
		n_dir_vector = np.cross(D1-D4,D3-D4)
		middle_of_bbox = 0.5*(D1+D3)
		lambda_1= distance/np.linalg.norm(n_dir_vector)
		sol1=middle_of_bbox+lambda_1*n_dir_vector #first solution for point
		lambda_2= -distance/np.linalg.norm(n_dir_vector)
		sol2=middle_of_bbox+lambda_2*n_dir_vector #second solution for point
		if np.linalg.norm(sol1)>np.linalg.norm(sol2):
			return list(sol2), correct_points_list #this point is closer to robot than the other solution
		else:
			return list(sol1), correct_points_list
	def can_robot_reach_target(self,pose_in_vectorspace):
		if pose_in_vectorspace[0:3]==[0,0,0]:
			return False, 0
		else:
			distance=round(np.linalg.norm(np.array(pose_in_vectorspace[0:3]-np.array([0,0,0.3]))),3)
			if distance>=self.robot_range_limit: #max range of Kinova Gen-3 ~ 0.9m
				# print("distance: ",distance)
				return False, round(self.robot_range_limit-distance,2)
			else:
				# print("distance: ",distance)
				return True, 0
	def can_robot_reach_inspection_pose(self,rectangle,dmap,current_joint_position,distance):
		xyz, plain_poses = self.get_pt_on_orthogonal_vector(rectangle,dmap,current_joint_position,distance)
		if xyz ==[0,0,0]:
			# print("depth was not calculated correctly")
			return False, 0, 0, plain_poses
		pose_mat = functions_.forward_kin(current_joint_position) #get 4x4 matrix from joint angles
		#pose_translation=list(pose_mat[:-1,3])
		rot_tait_bryan=list(R.from_matrix(pose_mat[:3,:3]).as_euler("zyx",degrees=True))
		inspection_pose = xyz + rot_tait_bryan

		distance=round(np.linalg.norm(np.array(xyz)),3)
		if distance>=self.robot_range_limit: #max range of Kinova Gen-3 ~ 0.9m
			return False, round(self.robot_range_limit-distance,2), inspection_pose, plain_poses
		else:
			return True, 0, inspection_pose, plain_poses



def posetrans(Startpose,Translation=[0,0,0],Rot=[0,0,0]):
	#works with scipy > 1.4.1
	if len(Translation)==3: #backwards compatibility
		trans_pose=Translation+Rot
	else:
		trans_pose=Translation #backwards compatibility
	start_pose=Startpose #backwards compatibility
	start_pose_rot_mat=R.from_euler("zyx",start_pose[3:6],degrees=True).as_matrix()
	start_pose_rot_mat=np.vstack([start_pose_rot_mat,[0,0,0]])
	start_pose_trans=np.array(start_pose[0:3]+[1])
	start_pose_trans=start_pose_trans.reshape(-1,1)
	start_pose_rot_mat=np.append(start_pose_rot_mat,start_pose_trans,axis=1)

	trans_pose_rot_mat=R.from_euler("zyx",trans_pose[3:6],degrees=True).as_matrix()
	trans_pose_rot_mat=np.vstack([trans_pose_rot_mat,[0,0,0]])
	trans_pose_trans=np.array(trans_pose[0:3]+[1])
	trans_pose_trans=trans_pose_trans.reshape(-1,1)
	trans_pose_rot_mat=np.append(trans_pose_rot_mat,trans_pose_trans,axis=1)

	new4x4=np.dot(start_pose_rot_mat,trans_pose_rot_mat)
	new_rot=R.from_matrix(new4x4[0:3,0:3]).as_euler("zyx",degrees=True)
	new_trans=new4x4[0:3,3]
	new_pose=list(new_trans)+list(new_rot)
	return new_pose


def image_alignment(dmap,dmap_img_cv,dmatfri):
	#the peth image and rgb image are not aligned
	#the parameters for the alignment were found by simple image overlay comparison
	# print("before: ",dmap)
	dmap=dmap[dmatfri[0]:dmatfri[1],dmatfri[2]:dmatfri[3]]
	# print("after: ",dmap)
	dmap_img_cv = dmap_img_cv[dmatfri[0]:dmatfri[1],dmatfri[2]:dmatfri[3]]
	return dmap,dmap_img_cv



def detect_objs_w_depth(cam_id,temporary_dict,CAM_ID,MAN_ID,KIN_ID,static_dict,
	w_shared,h_shared,delta_shared,current_joint_angles_shared,current_tcp_pose_shared,localhost):
	ButtonGenerator = functions_.Button()
	fontStyle = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.7
	fontThickness = 2
	depth_map_params_class=Depth_Map_Params()
	if camera=="Kinova":
		# 480 x 270 (16:9) @ 30, 15, 6 fps; FOV 72 ± 3° (diagonal)
		# original fov parameters from manual
		depth_map_params_class.fov_width = 62.75 #degrees
		depth_map_params_class.fov_height= 35.3 #degrees
		#is an alignment necessary? for kinova, yes.
		depth_map_params_class.alignment = True
		#y,x depth_map_area_to_fit_rgb_image
		#the parameters for the alignment were found by simple image overlay comparison #testpose [124.08, 293.52, 220.0, 263.33, 46.42, 315.74, 94.36]
		depth_map_params_class.dmatfri = [int(63*0.9),int(258*1.1),int(66-40),int(410+20)] #this is for an image of 270 x 480
		depth_map_params_class.alignment_offset_y = 135-((depth_map_params_class.dmatfri[1]-depth_map_params_class.dmatfri[0])/2+depth_map_params_class.dmatfri[0])
			# depth_map_params_class.dmatfri[0]+(270-
			# depth_map_params_class.dmatfri[1]) #270 init height
		depth_map_params_class.alignment_offset_x = (480-(depth_map_params_class.dmatfri[3]-
			depth_map_params_class.dmatfri[2]))/2-depth_map_params_class.dmatfri[2] #480 = init height
		#calculate alignment degrees
		depth_map_params_class.alignment_fov_angle_x_above = (depth_map_params_class.dmatfri[3]-240)*31.375/240
		depth_map_params_class.alignment_fov_angle_x_below = (240-depth_map_params_class.dmatfri[2])*31.375/240
		depth_map_params_class.alignment_fov_angle_y_above = (135-(270-depth_map_params_class.dmatfri[1]))*17.65/135
		depth_map_params_class.alignment_fov_angle_y_below = (135-depth_map_params_class.dmatfri[0])*17.65/135



		#TODO insert alignment parameters
	else: #Orbbec Astra S Pro
		depth_map_params_class.fov_width = 60 #degrees
		depth_map_params_class.fov_height= 49.5 #degrees
		#is an alignment necessary? for orbbec, no.
		depth_map_params_class.alignment = False

	class detection_set():
		def __init__(self):
			self.default_model_dir = './models'
			self.model = ""
			self.labels = ""
			self.threshold = 0.5
			self.engine = DetectionEngine(os.path.join(self.default_model_dir,self.model))
			self.labels = dataset_utils.read_label_file(os.path.join(self.default_model_dir,self.labels)) if self.labels else None


	# Start Edge TPU
	default_model_dir = './models'
	# default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
	default_model1 = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
	default_model2 = 'detect_elevator_panel_edgetpu.tflite'
	default_labels1 = 'coco_labels.txt'
	default_labels2 = 'elevator_panel_labels.txt'
	default_threshold = 0.1

	engine1 = DetectionEngine(os.path.join(default_model_dir,default_model1))
	labels1 = dataset_utils.read_label_file(os.path.join(default_model_dir,default_labels1)) if default_labels1 else None
	engine2 = DetectionEngine(os.path.join(default_model_dir,default_model2))
	labels2 = dataset_utils.read_label_file(os.path.join(default_model_dir,default_labels2)) if default_labels2 else None
	detection_sets = [
		['mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite','coco_labels.txt',0.1],
		['detect_elevator_panel_edgetpu.tflite','elevator_panel_labels.txt',0.5]
		]
	#Set up RGB Stream
	# Create full screen window
	screen_id = 0
	is_color = False
	# get the size of the screen

	depth_width, depth_height = 320*2 , 240*2 #320,240 are required values for orbbec camera
	rgb_width, rgb_height = 1920 , 1080 # only a trial to set the caption
	window_name = 'EdgeTPU_Detection'
	if camera=="Kinova":
		None
	else:
		# cam_id='rtsp://192.168.1.10/color'
		rgb_stream = cv2.VideoCapture(cam_id)
		# rgb_stream = WebcamVideoStream(cam_id).start()
		rgb_stream.set(cv2.CAP_PROP_FRAME_WIDTH, rgb_width)
		rgb_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, rgb_height)


	#Set up Depth Stream for Astra S or Kinova Cam
	if camera=="Kinova": #Settings to listen to ROS-Depth_map streamer
		HOST_depth='localhost'
		PORT_depth=8090
		#start multithreading Stream to fetch ROS images
		vs = receive_from_socket_stream_v2.Receive_Frame_And_Dmap()
		vs.start(HOST_depth,PORT_depth)
	else:
		if sys.platform == "linux" or sys.platform == "linux2":
			# linux
			dist = "./Redist_Linux"
		elif sys.platform == "darwin":
			# OS X
			dist = "./Redist_Linux"
		elif sys.platform == "win32":
			# Windows...
			dist = "./Redist"
		## Initialize openni and check
		openni2.initialize(dist)
		if (openni2.is_initialized()):
		    print("openNI2 initialized")
		else:
		    print("openNI2 not initialized")

		## Register the device
		dev = openni2.Device.open_any()

		## Create the depth stream
		depth_stream = dev.create_depth_stream()
		## Configure the depth_stream -- changes automatically based on bus speed
		depth_stream.set_video_mode(
		    c_api.OniVideoMode(
		        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
		        resolutionX=depth_width,
		        resolutionY=depth_height,
		        fps=15
		        )
		    )
		depth_stream.set_mirroring_enabled(False)
		depth_stream.start()
		time.sleep(1) #warm up the sensors

		def get_depth():
		    dmap = np.frombuffer(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(depth_height,depth_width)  # Works & It's FAST
		    dmap_img_cv = np.uint8(dmap.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
		    dmap_img_cv = cv2.cvtColor(dmap_img_cv,cv2.COLOR_GRAY2RGB)
		    # Shown unknowns in black
		    dmap_img_cv = 255 - dmap_img_cv
		    #dmap_img_cv = 255 - cv2.cvtColor(dmap_img_cv,cv2.COLOR_GRAY2RGB)
		    return dmap, dmap_img_cv

		def get_rgb(): #works for both cameras
			"""
			Returns numpy 3L ndarray to represent the rgb image.
			"""

			#bgr   = np.frombuffer(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,-1)
			ret, rgb_frame = rgb_stream.read()
			#rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
			return rgb_frame

	#Lets get the exact rgb_frame sizes
	if camera=="Kinova":
		success, dmap, dmap_img_cv, rgb_frame = vs.read()
		while success==False: #read until an image has been received from ROS
			success, dmap, dmap_img_cv, rgb_frame = vs.read()
		dmap,dmap_img_cv = image_alignment(dmap,dmap_img_cv,depth_map_params_class.dmatfri)
	else:
		dmap,dmap_img_cv = get_depth()
		rgb_frame = get_rgb()

	height_rgb, width_rgb = rgb_frame.shape[:2]
	depth_map_params_class.height_rgb, depth_map_params_class.width_rgb = rgb_frame.shape[:2]
	depth_map_params_class.height_dep, depth_map_params_class.width_dep = dmap_img_cv.shape[:2]
	depth_map_params_class.alignment_null_coordinate =[ #only for non aligned rgb and depth images
		int(depth_map_params_class.width_dep/2+depth_map_params_class.alignment_offset_x),
		int(depth_map_params_class.height_dep/2+depth_map_params_class.alignment_offset_y)] #x,y
	w_shared.value=width_rgb
	h_shared.value=height_rgb
	depth_map_params_class.ratio_height = height_rgb/depth_map_params_class.height_dep
	depth_map_params_class.ratio_width = width_rgb/depth_map_params_class.width_dep
	vision_middle=(int(width_rgb/2),int(height_rgb/2)-150)
	print("Depth: height: ",depth_map_params_class.height_dep," width: ",depth_map_params_class.width_dep)
	print("Depth: fov_height: ",depth_map_params_class.fov_height," fov_width: ",depth_map_params_class.fov_width)
	print("RGB: height: ",height_rgb," width: ",width_rgb)

	#init framerate display
	t0 = t00 = t000 = time.time()
	frame_count=0
	framerate=0

	token=0 #will be used for display timer
	start_token=1 #for the Videostream
	start_token_thread=1 # for the initial thread


	#stream settings for GUI stream
	if localhost==True:
		HOST="localhost"#"172.20.10.2"#"raspberrypi"#'10.0.1.230'#'172.20.10.2'#'192.168.0.19'#'localhost'
	else:
		# HOST="172.20.10.2"
		HOST="192.168.168.125"
	PORT=8089
	# sender_thread=receive_from_socket_stream_v2.Send_Frame_And_Dict().start(HOST,PORT)
	gui_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	gui_client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

	gui_client_socket.connect((HOST, PORT))
	connection = gui_client_socket.makefile('wb')
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
	def gui_sender(rgb_frame,temporary_dict_copy):
		#things to try:
		# print(type(temporary_dict_copy))
		print("sending frame")
		t0=time.time()
		result, rgb_frame = cv2.imencode('.jpg', rgb_frame, encode_param)
		# result, dmap_img_cv = cv2.imencode('.jpg', dmap_img_cv, encode_param)
		frame_dict={"imageFile":rgb_frame,"buttonDict":temporary_dict_copy}
		data = pickle.dumps(frame_dict,fix_imports=True)
		size = len(data)
		gui_client_socket.sendall(struct.pack("L", size) + data)
		print("Frame sent after ",round(time.time()-t0,5)," seconds")
		return

	def write_to_temporary_dict(objs):
		detected_labels=[]  # this list is used to clear the temp_dict of /
		# objects that have not been found
		if objs:
			#scale bbox to depth map, evaluate and show in image
			for obj in objs:
				# x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
				x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
				x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
				target_middle=(int(x0+(x1-x0)/2),int(y0+(y1-y0)/2))
				# read the depth and calculate the pose in vetorspace
				pose_in_vectorspace=depth_map_params_class.px_point2xyz(target_middle,dmap,current_joint_angles_shared)
				distance_obj=depth_map_params_class.distance_obj
				# cv2.circle(rgb_frame,(target_middle[0],target_middle[1]),3,(255,255,0),1)
				# cv2.circle(dmap_img_cv,(target_middle_depth[0],target_middle_depth[1]),3,(255,255,0),1)
				# cv2.putText(rgb_frame, str(distance_obj)+" m", (target_middle[0]-10,target_middle[1]-10),fontStyle, fontScale, (0, 0, 255), fontThickness)
				#update shared data
				obj_label=labels[obj.label_id]
				detected_labels.append(obj_label)
				#Calculate the vectorspace pose
				# if distance_obj!=0.0:
				# 	cv2.putText(rgb_frame, str(pose_in_vectorspace[0:3])+" m", (target_middle[0]-10,target_middle[1]+20),fontStyle, fontScale, (0, 0, 255), 2)

				temporary_dict[obj_label] = {
					"rectangle" : [x0,y0,x1,y1],
					"target_middle" : target_middle,
					"depth" : distance_obj,
					"vector_space" : pose_in_vectorspace,
					"on_click" : obj_label, #ID's are found in a look-up table
					}
		return

	def spin_up_engine(input):
		detected_labels=[]
		engine,labels,pil_im,default_threshold,exclude_list = input
		objs = engine.detect_with_image(pil_im,
										threshold=default_threshold,
										keep_aspect_ratio='store_true',
										relative_coord=False,
										top_k=10)
		if objs:
			for obj in objs:
				obj_label=labels[obj.label_id]
				# x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
				x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
				x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
				if x1-x0>width_rgb/2 or obj_label in exclude_list:
					objs.remove(obj)
					continue #the bounding box is too big, must be false-positive
				else:
					detected_labels.append(obj_label)
		return objs,labels,detected_labels
	que = Queue()
	#####################################################
	# import concurrent.futures
	# import urllib.request
	# #args=(que, engine1,labels1,pil_im,default_threshold=0.5)
	# """URLS = ['http://www.foxnews.com/',
	#         'http://www.cnn.com/',
	#         'http://europe.wsj.com/',
	#         'http://www.bbc.co.uk/',
	#         'http://some-made-up-domain.com/']"""
	#
	# # Retrieve a single page and report the URL and contents
	# """def load_url(url, timeout):
	#     with urllib.request.urlopen(url, timeout=timeout) as conn:
	#         return conn.read()"""
	# # def spin_up_engine
	#
	# # We can use a with statement to ensure threads are cleaned up promptly
	# with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
	#     # Start the load operations and mark each future with its URL
	#     future_to_url = {executor.submit(spin_up_engine, url, 60): url for url in URLS}
	#     for future in concurrent.futures.as_completed(future_to_url):
	#         url = future_to_url[future]
	#         try:
	#             data = future.result()
	#         except Exception as exc:
	#             print('%r generated an exception: %s' % (url, exc))
	#         else:
	#             print('%r page is %d bytes' % (url, len(data)))
	#####################################################
	### MAIN LOOP ###
	while [CAM_ID[0],CAM_ID[1]] != [0,0]: #[0,0] ends the program
		try:
			# temporary_dict.clear()
			# rgb_frame.clear()
			if camera=="Kinova":
				# print("get image")
				success, dmap, dmap_img_cv, rgb_frame_stream = vs.read()
				dmap,dmap_img_cv = image_alignment(dmap,dmap_img_cv,depth_map_params_class.dmatfri)
				# print("got image")
			else:
				dmap,dmap_img_cv = get_depth()
				rgb_frame_stream = get_rgb()
			rgb_frame=rgb_frame_stream.copy()
			pil_im = Image.fromarray(rgb_frame.copy())

			#Start the threads
			t1 = Thread(
				target=lambda q,
				arg1: q.put(spin_up_engine(arg1)),
				args=(que,[engine1,labels1,pil_im,0.2,[]])
				)
			t1.start()
			t2 = Thread(
				target=lambda q,
				arg1: q.put(spin_up_engine(arg1)),
				# args=(que, [engine2,labels2,pil_im,0.5,["Elevator_Panel1","Call_Elevator1"]])
				args=(que, [engine2,labels2,pil_im,0.5,["Elevator_Panel2","Call_Elevator2",
														"Elevator_Panel3","Call_Elevator3",]])
				)
			t2.start()
			t1.join()
			t2.join()

			#Quick Fix, remove when not needed: detect contours in direct Mode
			if [CAM_ID[0],CAM_ID[1]] == [3,5]:
				# Click Mode Button
				rgb_frame, __ = button_cv_final.find_buttons_in_bbox(rgb_frame, [],whole_frame=True)
			###################################

			# Check thread's return value
			detected_labels_all=[]

			while not que.empty():
				#Retreive from threding queue
				objs,labels,detected_labels = que.get()
				#button/Contour detector
				rgb_frame, __ = button_cv_final.find_buttons_in_bbox(rgb_frame, objs)
				# draw objects
				# rgb_frame = detect.append_objs_to_img(rgb_frame, objs, labels)
				write_to_temporary_dict(objs) # write to temporary dict
				detected_labels_all+=detected_labels
			#now clear the temp_dict of elements that have not been found
			# elements_to_remove = [x for x in [labels[x] for x in list(labels.keys())] if x not in detected_labels]
			# #remove everything that has not been detected in current frame
			# [temporary_dict.pop(key, None) for key in elements_to_remove]
			# print("temporary_dict: ",json.dumps(dict(temporary_dict),sort_keys=True, indent=4))
			# print("detected_labels_all ",detected_labels_all)
			for elem in dict(temporary_dict):
				if elem not in detected_labels_all:
					# print("i must remove ",elem)
					temporary_dict.pop(elem, None)
			# print("temporary_dict: ",json.dumps(dict(temporary_dict),sort_keys=True, indent=4))

			# objs1 = spin_up_engine(engine1,labels1,pil_im,default_threshold=0.5)
			# objs2 = spin_up_engine(engine2,labels2,pil_im,default_threshold=0.5,
			# 	exclude_list=["Elevator_Panel1","Call_Elevator1"])

			"""#button/Contour detector
			rgb_frame, button_bbox_all = button_cv_final.find_buttons_in_bbox(rgb_frame, objs1)
			rgb_frame, button_bbox_all = button_cv_final.find_buttons_in_bbox(rgb_frame, objs2)
			#TODO draw objects
			rgb_frame = detect.append_objs_to_img(rgb_frame, objs1, labels1)
			rgb_frame = detect.append_objs_to_img(rgb_frame, objs2, labels2)
			##################
			# write to temporary dict
			write_to_temporary_dict(objs1)
			write_to_temporary_dict(objs2)


			# engine1
			#now clear the temp_dict of elements that have not been found
			elements_to_remove = [x for x in [labels1[x] for x in list(labels1.keys())] if x not in detected_labels]
			#remove everything that has not been detected in current frame
			[temporary_dict.pop(key, None) for key in elements_to_remove]

			# engine2
			#now clear the temp_dict of elements that have not been found
			elements_to_remove = [x for x in [labels1[x] for x in list(labels1.keys())] if x not in detected_labels]
			#remove everything that has not been detected in current frame
			[temporary_dict.pop(key, None) for key in elements_to_remove]"""

			"""
				# #SECOND ENGINE
				# objs2 = engine2.detect_with_image(pil_im,
			    #                                 threshold=default_threshold,
			    #                                 keep_aspect_ratio='store_true',
			    #                                 relative_coord=False,
			    #                                 top_k=10)
				# # print(objs2)
				#
				# #get the current TCP Pose
				# # current_position=[0,0,0,0,0,0] #testing, must come from KINEMATIC
				# detected_labels=[]  # this list is used to clear the temp_dict of /
				# # objects that have not been found
				# filtered_objs=[]
				# if objs2:
				# 	#scale bbox to depth map, evaluate and show in image
				# 	for obj in objs2:
				# 		# x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
				# 		# print("obj: ",obj)
				# 		x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
				# 		x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
				# 		obj_label=labels2[obj.label_id]
				# 		# print(obj_label,": ",obj.score," %")
				# 		if not (x1-x0>width_rgb/2 or "Elevator_Panel1" in obj_label or "Call_Elevator1" in obj_label):
				# 			filtered_objs.append(obj)
				# 			detected_labels.append(obj_label)
				# 			# objs2.remove(obj)
				# 			# del objs2.obj
				# 			# delattr(objs2, obj)
				# 			# continue #the bounding box is too big, must be false-positive
				# 		else:
				# 			# filtered_objs.append(obj)
				# 			None
				# 	#Testing button detector
				# 	rgb_frame, button_bbox_all = button_cv_final.find_buttons_in_bbox(rgb_frame, filtered_objs)
				# 	for obj in filtered_objs:
				# 		x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
				# 		x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
				# 		obj_label=labels2[obj.label_id]
				# 		target_middle=(int(x0+(x1-x0)/2),int(y0+(y1-y0)/2))
				# 		# read the depth and calculate the pose in vetorspace
				# 		pose_in_vectorspace=depth_map_params_class.px_point2xyz(target_middle,dmap,current_joint_angles_shared)
				# 		distance_obj=depth_map_params_class.distance_obj
				# 		cv2.circle(rgb_frame,(target_middle[0],target_middle[1]),3,(255,255,0),1)
				# 		# cv2.circle(dmap_img_cv,(target_middle_depth[0],target_middle_depth[1]),3,(255,255,0),1)
				# 		cv2.putText(rgb_frame, str(distance_obj)+" m", (target_middle[0]-10,target_middle[1]-10),fontStyle, fontScale, (0, 0, 255), fontThickness)
				# 		#update shared data
				# 		# obj_label=labels[obj.label_id]
				#
				#
				# 		#Calculate the vectorspace pose
				# 		if distance_obj!=0.0:
				# 			cv2.putText(rgb_frame, str(pose_in_vectorspace[0:3])+" m", (target_middle[0]-10,target_middle[1]+20),fontStyle, fontScale, (0, 0, 255), 2)
				#
				# 		temporary_dict[obj_label] = {
				#             "rectangle" : [x0,y0,x1,y1],
				#             "target_middle" : target_middle,
				#             "depth" : distance_obj,
				#             "vector_space" : pose_in_vectorspace,
				#             "on_click" : obj_label, #ID's are found in a look-up table
				#             }
				#
				# 	cv2_im = detect.append_objs_to_img(cv2_im, filtered_objs, labels2)
				# # elements_to_remove = [[labels2[x] for x in list(labels2.keys())].remove(detected) for detected in detected_labels]
				# # elements_to_remove = [list(labels2.keys()).remove(detected) for detected in detected_labels]
				# # elements_to_remove = [x for x in list(labels2.keys()) if x not in elements_to_remove]
				# elements_to_remove = [x for x in [labels2[x] for x in list(labels2.keys())] if x not in detected_labels]
				# #remove everything that has not been detected in current frame
				# [temporary_dict.pop(key, None) for key in elements_to_remove]

				# print("detected_labels: ",detected_labels) #5
				# print("list(labels2.keys()): ",list(labels2.keys())) #[1,2,3,4,5]
				# print("removing keys: ",elements_to_remove) #[1,2,3,4] but None"""


			#annotate image
			cv2.putText(rgb_frame, "Framerate: "+str(framerate), (10,15),fontStyle, 0.5, (0, 0, 255), 2)
			distance_middle=dmap[int(depth_map_params_class.height_dep/2),int(depth_map_params_class.width_dep/2)]/1000
			if distance_middle!=0.0:
				pose_in_vectorspace=depth_map_params_class.px_point2xyz(
					(int(width_rgb/2),int(height_rgb/2)),dmap,current_joint_angles_shared)
				cv2.putText(rgb_frame, str(pose_in_vectorspace[0:3])+" m", (int(width_rgb/2)-10,int(height_rgb/2)+20),fontStyle, fontScale, (0, 0, 255), 2)
			# cv2.circle(dmap_img_cv,(int(depth_map_params_class.width_dep/2),int(depth_map_params_class.height_dep/2)),3,(255,255,0),1)
			cv2.circle(rgb_frame,(int(width_rgb/2),int(height_rgb/2)),3,(255,255,0),1)
			cv2.putText(rgb_frame, str(distance_middle)+" m", (int(width_rgb/2)-10,int(height_rgb/2)-10),fontStyle, fontScale, (0, 0, 255), fontThickness)
			#More Points
			# temp_point=(int(1*width_rgb/4),int(1*height_rgb/4))
			# pose_in_vectorspace=depth_map_params_class.px_point2xyz(temp_point,dmap,current_joint_angles_shared)
			# cv2.putText(rgb_frame, str(pose_in_vectorspace[0:3])+" m", (temp_point[0]-10,temp_point[1]+20),fontStyle, fontScale, (0, 0, 255), 2)
			# cv2.circle(rgb_frame,(temp_point[0],temp_point[1]),3,(255,255,0),1)
			#
			# temp_point=(int(1*width_rgb/4),int(3*height_rgb/4))
			# pose_in_vectorspace=depth_map_params_class.px_point2xyz(temp_point,dmap,current_joint_angles_shared)
			# cv2.putText(rgb_frame, str(pose_in_vectorspace[0:3])+" m", (temp_point[0]-10,temp_point[1]+20),fontStyle, fontScale, (0, 0, 255), 2)
			# cv2.circle(rgb_frame,(temp_point[0],temp_point[1]),3,(255,255,0),1)
			#
			# temp_point=(int(3*width_rgb/4),int(1*height_rgb/4))
			# pose_in_vectorspace=depth_map_params_class.px_point2xyz(temp_point,dmap,current_joint_angles_shared)
			# cv2.putText(rgb_frame, str(pose_in_vectorspace[0:3])+" m", (temp_point[0]-10,temp_point[1]+20),fontStyle, fontScale, (0, 0, 255), 2)
			# cv2.circle(rgb_frame,(temp_point[0],temp_point[1]),3,(255,255,0),1)
			#
			# temp_point=(int(3*width_rgb/4),int(3*height_rgb/4))
			# pose_in_vectorspace=depth_map_params_class.px_point2xyz(temp_point,dmap,current_joint_angles_shared)
			# cv2.putText(rgb_frame, str(pose_in_vectorspace[0:3])+" m", (temp_point[0]-10,temp_point[1]+20),fontStyle, fontScale, (0, 0, 255), 2)
			# cv2.circle(rgb_frame,(temp_point[0],temp_point[1]),3,(255,255,0),1)

			# Abort Button
			rgb_frame, temporary_dict = ButtonGenerator.insert_button(
		                rgb_frame, temporary_dict, "Abort",0,"top_left")
			# Quit Button
			rgb_frame, temporary_dict = ButtonGenerator.insert_button(
		                rgb_frame, temporary_dict, "Quit",2,"top_right")
			#Move Robot Button
			rgb_frame, temporary_dict = ButtonGenerator.insert_button(
		                rgb_frame, temporary_dict, "CMD",0,"bottom_left")

			#Application states for dynamic GUI creation
			if [CAM_ID[0],CAM_ID[1]] == [1,0]:
				cv2.putText(rgb_frame, "Click on Box", (10,height_rgb-60),fontStyle, fontScale, (0, 0, 255), fontThickness)
				token33=1
				token36=1
				token39=1
				# Click Mode Button (direct)
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Tap2Touch",1,"top_left")
				# Click Mode Button (orthogonal)
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Tap2Align",2,"top_left")
			if ([CAM_ID[0],CAM_ID[1]] == [2,0] or [CAM_ID[0],CAM_ID[1]] == [3,0] or [CAM_ID[0],CAM_ID[1]] == [3,3]) and static_dict["chosen_object"] in temporary_dict: #object detected
				cv2.putText(rgb_frame, "Aiming at target "+static_dict["chosen_object"], (10,height_rgb-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
				target_mid=temporary_dict[static_dict["chosen_object"]]["target_middle"]
				delta_shared[0]=target_mid[0]-vision_middle[0]
				delta_shared[1]=target_mid[1]-vision_middle[1]
				cv2.line(rgb_frame,vision_middle,target_mid,(0,255,0),1)


			if [CAM_ID[0],CAM_ID[1]] == [3,0]:
				# Follow Button
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Follow",1,"top_left")
				# Inspect Button
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Inspect",2,"top_left")
				# Click Mode Button
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Tap2Touch",3,"top_left")

			if ([CAM_ID[0],CAM_ID[1]] == [3,1] or
				[CAM_ID[0],CAM_ID[1]] == [3,3]) and (
					static_dict["chosen_object"] in temporary_dict):
				#TODO add try except for catching depth=0.0 Exception (uncomment in px_point2xyz)
				can_reach_target, remaining_dist_target = depth_map_params_class.can_robot_reach_target(
					temporary_dict[static_dict["chosen_object"]]["vector_space"]) #pose in vectorspace


				#Check if robot can reach an orthogonal pose in front of the target
				#(Plain must be calculated)
				x00, y00, x11, y11 = temporary_dict[static_dict["chosen_object"]]["rectangle"]
				x_padding=int((x1-x0)/4)
				y_padding=int((y1-y0)/4)
				rectangle = [x00+x_padding, y00+y_padding, x11-x_padding, y11-y_padding]
				cv2.rectangle(rgb_frame,(rectangle[0],rectangle[1]),(rectangle[2],rectangle[3]),(0,0,255),2)
				distance = 0.3 # inspection pose, meters before object
				# try: # (sometimes depth is 0)
				can_reach_inspect, remaining_dist_inspect, inspection_pose, plain_poses = \
					depth_map_params_class.can_robot_reach_inspection_pose(
						rectangle,dmap,current_joint_angles_shared,distance)
				# except:
				# 	# print("not enough valid points to calculate plane")
				# 	cv2.putText(rgb_frame,"depth was not meassured correctly, please move arm" , (10,height_rgb-90-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
				# 	can_reach_inspect, remaining_dist_inspect, inspection_pose = False,0,0
				"""
				Moving to inspection pose still needs the Orientation calculated
				if can_reach_target==True and can_reach_inspect==True:
					# temporary_dict[static_dict["chosen_object"]]["inspection_pose"]=inspection_pose
					static_dict["inspection_pose"]=inspection_pose
					print("Target can be reached")
					if token33==1:
						MAN_ID[0],MAN_ID[1] = [3,1] #move robot to orthogonal pose
						token33=0
				"""
				if can_reach_target==True:
					print("Tapping the button!")
					if token33==1:
						static_dict["start_joint_pose"]=list(current_joint_angles_shared)
						# pose_mat = functions_.forward_kin(current_joint_angles_shared) #get 4x4 matrix from joint angles
						# #pose_translation=list(pose_mat[:-1,3])
						# target_rot_tait_bryan=list(R.from_matrix(pose_mat[:3,:3]).as_euler("zyx",degrees=True))
						target_xyz=temporary_dict[static_dict["chosen_object"]]["vector_space"]
						static_dict["target_position"]=target_xyz[0:3]+current_tcp_pose_shared[3:6]
						print("static_dict[target_position]: ", static_dict["target_position"])
						time.sleep(0.1)
						MAN_ID[0],MAN_ID[1] = [3,4] #tap button
						token33=0
				elif can_reach_inspect == False and remaining_dist_inspect == 0 and inspection_pose == 0:
					cv2.putText(rgb_frame, "Inspection pose can not be calculated because depth map has blind spots in the area", (10,height_rgb-30-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
				else:
					cv2.putText(rgb_frame, static_dict["chosen_object"]+" out of reach, please move the robot "+str(remaining_dist_target)+" meters closer to target", (10,height_rgb-60-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
					cv2.putText(rgb_frame, static_dict["chosen_object"]+" out of reach, please move the robot "+str(remaining_dist_inspect)+" meters closer to inspection position", (10,height_rgb-30-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
					# if token==0:
					# 	t1=time.time()
					# 	token=1
					# if time.time()-t1>4:
					# 	MAN_ID[0],MAN_ID[1] = [3,0] #back to "inspect" menu
					# 	token=0

			if [CAM_ID[0],CAM_ID[1]] == [3,1] and static_dict["chosen_object"] in temporary_dict:
				cv2.putText(rgb_frame, "Moving arm into inspecton position", (10,height_rgb-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
				# rectangle = temporary_dict[static_dict["chosen_object"]]["rectangle"]
				# distance = 0.3 #meters before object
				#TODO OPTIONAL reconstruct inspection pose into px-coordinates?
				#TODO in KIN retrieve inspection_pose from buttonDict?
			if [CAM_ID[0],CAM_ID[1]] == [3,4] and static_dict["chosen_object"] in temporary_dict:
				# Click Mode Button
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Tap2Touch",1,"top_left")
				cv2.putText(rgb_frame, static_dict["chosen_object"]+" in reach, activate 'Click Mode' to tap a button", (10,height_rgb-30-30),fontStyle, fontScale, (0, 0, 255), fontThickness)

			if [CAM_ID[0],CAM_ID[1]] == [3,5]:
				# Click Mode Button
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Tap2Touch",1,"top_left",highlighted=True)
				cv2.putText(rgb_frame, "Click Mode: Click any Pixel to make the robot tap that point", (10,height_rgb-30-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
				# rgb_frame, __ = button_cv_final.find_buttons_in_bbox(rgb_frame, [],whole_frame=True)

			if [CAM_ID[0],CAM_ID[1]] == [3,6]:
				# Evaluate Reachability of Clicked Position
				# convert clicked_point (px) to Vector space coordinates XYZ
				clicked_point = static_dict["clicked_point"]
				pose_in_vectorspace=depth_map_params_class.px_point2xyz(clicked_point,dmap,current_joint_angles_shared)
				# distance_obj=depth_map_params_class.distance_obj
				#Check if pose can be reached by robot
				can_reach_target, remaining_dist_target = depth_map_params_class.can_robot_reach_target(pose_in_vectorspace) #pose in vectorspace
				if can_reach_target==True:
					print("Tapping the button!")
					if token36==1:
						static_dict["start_joint_pose"]=list(current_joint_angles_shared)
						# pose_mat = functions_.forward_kin(current_joint_angles_shared) #get 4x4 matrix from joint angles
						# #pose_translation=list(pose_mat[:-1,3])
						# target_rot_tait_bryan=list(R.from_matrix(pose_mat[:3,:3]).as_euler("zyx",degrees=True))
						target_xyz=pose_in_vectorspace
						static_dict["target_position"]=target_xyz[0:3]+current_tcp_pose_shared[3:6]
						print("static_dict[target_position]: ", static_dict["target_position"])
						time.sleep(0.1)
						MAN_ID[0],MAN_ID[1] = [3,7] #tap button
						token36=0
				else:
					cv2.putText(rgb_frame, "Clicked point out of reach, please move the robot "+str(remaining_dist_target)+" meters closer to target", (10,height_rgb-60-30),fontStyle, fontScale, (0, 0, 255), fontThickness)

			if [CAM_ID[0],CAM_ID[1]] == [3,7]:
				# Click Mode Button
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Tap2Touch",1,"top_left",highlighted=True)
				#TODO find a point that is close to the XYZ pose within dmap to show
				#dynamically the pose that will be reached (involves tracking algorithm)
				cv2.putText(rgb_frame, "Moving robot to tap point "+str(pose_in_vectorspace),
					(10,height_rgb-30-30),fontStyle, fontScale, (0, 0, 255), fontThickness)

			if [CAM_ID[0],CAM_ID[1]] == [3,8]:
				# Click Mode Button
				# Click Mode Button (orthogonal)
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "Tap2Align",2,"top_left",highlighted=True)
				cv2.putText(rgb_frame, "Click Mode: Click any Pixel to make the robot tap that point",
					(10,height_rgb-30-30),fontStyle, fontScale, (0, 0, 255), fontThickness)

			if [CAM_ID[0],CAM_ID[1]] == [3,9]:
				# Evaluate Reachability of Clicked Position (Orthogonal)
				clicked_point = static_dict["clicked_point"]
				cv2.circle(rgb_frame,(clicked_point[0],clicked_point[1]),3,(0,0,255),1)
				pose_in_vectorspace=depth_map_params_class.px_point2xyz(clicked_point,
					dmap,current_joint_angles_shared)
				#TODO add try except for catching depth=0.0 Exception (uncomment in px_point2xyz)
				can_reach_target, remaining_dist_target = depth_map_params_class.can_robot_reach_target(
					pose_in_vectorspace) #pose in vectorspace
				if can_reach_target==True:
					#Check if robot can reach an orthogonal pose in front of the target
					#(Plain must be calculated)

					# use the vector of the orthogonal pose for the middle point
					# for the clicked point
					pose_in_vectorspace_middle=depth_map_params_class.px_point2xyz(
						(int(width_rgb/2),int(height_rgb/2)),dmap,current_joint_angles_shared)

					delta_x = delta_y = int(1/32*width_rgb) #distance from point in pixels
					rectangle_clicked =[clicked_point[0]-delta_x,
								clicked_point[1]-delta_y,
								clicked_point[0]+delta_x,
								clicked_point[1]+delta_y]
					rectangle =[int(width_rgb/2)-delta_x, #rectangle of middle point
								int(height_rgb/2)-delta_y, # because of distortion
								int(width_rgb/2)+delta_x, #
								int(height_rgb/2)+delta_y] #
					cv2.rectangle(rgb_frame,(rectangle_clicked[0],rectangle_clicked[1]),
						(rectangle_clicked[2],rectangle_clicked[3]),(0,0,255),1)
					distance = 0.2 # inspection pose, meters before object
					#Now use the pixel rectangle to calculate an orthogonal pose in front of the surface
					can_reach_inspect, remaining_dist_inspect, inspection_pose, plain_poses = \
						depth_map_params_class.can_robot_reach_inspection_pose(
							rectangle,dmap,current_joint_angles_shared,distance)
					vec_mid_to_inspect = list(map(sub,
						inspection_pose,pose_in_vectorspace_middle)) #(x,y,z)
					#calculate the inspection pose depending on middle insp-pose
					inspection_pose = list(map(add,
						pose_in_vectorspace,vec_mid_to_inspect))
					#inspection_pose has not yet the correct orientation
					if can_reach_inspect==True or 1==1:
						#Calculate orientation
						print("DEBUGGING: point on wall ",pose_in_vectorspace[0:3])
						print("DEBUGGING: orthogonal point ",inspection_pose[0:3])
						rot_mat=orientation_regulator_final.calc_rotmat_to_align_vec_a_to_vec_b(
							pose_in_vectorspace[0:3],inspection_pose[0:3])
						#Convert the rotation matrix back to euler in the sequence of XYZ
						# Why not zyx like in the Kinova Manual?
						# Because this did not work.
						rot_tait_bryan=list(R.from_matrix(rot_mat).as_euler("XYZ",degrees=True))

						#For Testing Purposes

						if token39==1:
							print("Moving to inspection pose!")
							static_dict["start_joint_pose"]=list(current_joint_angles_shared)
							# pose_mat = functions_.forward_kin(current_joint_angles_shared) #get 4x4 matrix from joint angles
							# #pose_translation=list(pose_mat[:-1,3])
							# target_rot_tait_bryan=list(R.from_matrix(pose_mat[:3,:3]).as_euler("zyx",degrees=True))
							target_pose_round=[round(p,4) for p in inspection_pose[0:3]+rot_tait_bryan]
							static_dict["target_position"]=target_pose_round
							print("static_dict[target_position]: ", static_dict["target_position"])
							time.sleep(0.1)
							MAN_ID[0],MAN_ID[1] = [3,10] #Move to inspect pose
							token39=0
					else:
						cv2.putText(rgb_frame, "Inspect pose out of reach, please move the robot "+str(remaining_dist_inspect)+" meters closer ", (10,height_rgb-60-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
				else:
					cv2.putText(rgb_frame, "Clicked point out of reach, please move the robot "+str(remaining_dist_target)+" meters closer ", (10,height_rgb-60-30),fontStyle, fontScale, (0, 0, 255), fontThickness)

			if [CAM_ID[0],CAM_ID[1]] == [3,10]:
				# Click Mode Button
				#TODO find a point that is close to the XYZ pose within dmap to show
				#dynamically the pose that will be reached (involves tracking algorithm)
				cv2.putText(rgb_frame, "Moving robot to inspection pose "+str(pose_in_vectorspace),
					(10,height_rgb-30-30),fontStyle, fontScale, (0, 0, 255), fontThickness)

			if [CAM_ID[0],CAM_ID[1]] == [4,0] and static_dict["chosen_object"] in temporary_dict:
				#(msg:detect buttons / clickable elements + option to move back and forth)
				#INSERT function to detect buttons, get xyz-pose and write clickable button dict
				cv2.putText(rgb_frame, "Click an object to interact", (10,height_rgb-30),fontStyle, fontScale, (0, 0, 255), fontThickness)
				# Move Away
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "move_away",1,"top_left")
				# Move Closer
				rgb_frame, temporary_dict = ButtonGenerator.insert_button(
			                rgb_frame, temporary_dict, "move_closer",2,"top_left")
			if [CAM_ID[0],CAM_ID[1]] == [4,3]:
				cv2.putText(rgb_frame, "Moving to target position", (10,height_rgb-30),fontStyle, fontScale, (0, 0, 255), fontThickness)

			if [CAM_ID[0],CAM_ID[1]] == [5,0]: #draw movement possibilities
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=None)
			elif [CAM_ID[0],CAM_ID[1]] == [5,1]: #right
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=2)
			elif [CAM_ID[0],CAM_ID[1]] == [5,2]: #left
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=0)
			elif [CAM_ID[0],CAM_ID[1]] == [5,3]: #up
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=4)
			elif [CAM_ID[0],CAM_ID[1]] == [5,4]: #down
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=1)
			elif [CAM_ID[0],CAM_ID[1]] == [5,5]: #rot -90 deg
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=3)
			elif [CAM_ID[0],CAM_ID[1]] == [5,6]: #rot +90 deg
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=5)
			elif [CAM_ID[0],CAM_ID[1]] == [5,7]: #forward
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=6)
			elif [CAM_ID[0],CAM_ID[1]] == [5,8]: #backward
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=8)
			elif [CAM_ID[0],CAM_ID[1]] == [5,9]: #HOME
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=7)
			elif [CAM_ID[0],CAM_ID[1]] == [5,10]: #TRANSPORT
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=9)
			elif [CAM_ID[0],CAM_ID[1]] == [5,11]: #HOME-revers
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=10)
			elif [CAM_ID[0],CAM_ID[1]] == [5,12]: #
				rgb_frame, temporary_dict = ButtonGenerator.insert_arrows(rgb_frame, temporary_dict,highlighted=11)


			#send image to gui client
			# cv2.putText(rgb_frame, "MAN_ID: "+str(MAN_ID[0])+"."+str(MAN_ID[1]), (width_rgb-200,30),fontStyle, fontScale, (0,0,255), fontThickness)
			# cv2.putText(rgb_frame, "KIN_ID: "+str(KIN_ID[0])+"."+str(KIN_ID[1]), (width_rgb-200,60),fontStyle, fontScale, (0,0,255), fontThickness)
			# cv2.putText(rgb_frame, "CAM_ID: "+str(CAM_ID[0])+"."+str(CAM_ID[1]), (width_rgb-200,90),fontStyle, fontScale, (0,0,255), fontThickness)

			# scale_dep = imutils.resize(dmap_img_cv, width=width_rgb)

			# img is a numpy array/cv2 image
			img = dmap_img_cv.copy()
			img = img - img.min() # Now between 0 and 8674
			img = img / img.max() * 255
			# Now that the image is between 0 and 255, we can convert to a 8-bit integer.
			new_img = np.uint8(img)
			backtorgb = cv2.cvtColor(new_img,cv2.COLOR_GRAY2RGB)
			# Now merge the depth image into the rgb image
			rgb_frame[-backtorgb.shape[0]::, -backtorgb.shape[1]::] = backtorgb
			# cv2.imshow('depth_temp', backtorgb)
			cv2.imshow(window_name, rgb_frame)
			cv2.waitKey(1)
			# cv2.imwrite( "Depth_Image.png", scale_dep )
			# print(type(dmap_img_cv))
			temporary_dict_copy=dict(temporary_dict)
			# # print(type(temporary_dict_copy))
			# result, rgb_frame = cv2.imencode('.jpg', rgb_frame, encode_param)
			# # result, dmap_img_cv = cv2.imencode('.jpg', dmap_img_cv, encode_param)
			# frame_dict={"imageFile":rgb_frame,"buttonDict":temporary_dict_copy}

			# sender_thread.send(frame_dict)
			if start_token_thread==1: #start the first thread with the first frame
				t = Thread(target=gui_sender,args=(rgb_frame,temporary_dict_copy))
				t.start()
				start_token_thread=0
			elif not t.is_alive() and time.time()-t00 > 1./10.: #when thread has finished, allow to start a new thread
				t = Thread(target=gui_sender,args=(rgb_frame,temporary_dict_copy))
				t.start()
				t00 = time.time() # the time delay is neccessary in a weak network


			# data = pickle.dumps(frame_dict,fix_imports=True)
			# #alternate pickle version
			#
			# size = len(data)
			# gui_client_socket.sendall(struct.pack("L", size) + data)
			if record_video==True and time.time()-t000 > 1./10.:
				if start_token==1:
					#Init the frame recorder
					date_string=datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
					out = cv2.VideoWriter('./Recordings/CAM_rec_%s.avi'%(date_string),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width_rgb, height_rgb))
					start_token=0
				out.write(rgb_frame)
				t000 = time.time()



			t1=time.time()
			frame_count+=1
			if t1-t0>=1.0:
			    framerate=frame_count
			    frame_count=0
			    t0=time.time()
			# while time.time()-t00 < 1./10.:
			# 	time.sleep(0.003)
			# t00 = time.time()

		except Exception as e:
			print(e)
			MAN_ID[0],MAN_ID[1] = [0,0]

	if camera=="Kinova":
		vs.stop()
	else:
		rgb_stream.release()
		depth_stream.stop()
		openni2.unload()
	sender_thread.stop()
	out.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	manager=multiprocessing.Manager()
	temporary_dict=manager.dict()

	detect_objs_w_depth(cam_id=1,temporary_dict=temporary_dict)
