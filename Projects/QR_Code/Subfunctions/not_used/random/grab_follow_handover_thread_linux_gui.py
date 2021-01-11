# Lint as: python3
from Subfunctions import functions_
from Subfunctions import highlevel_movements
from Subfunctions import vision_action
import argparse
import time
import datetime
import os
import signal
from PIL import Image
from PIL import ImageDraw
import cv2
from Subfunctions import detect
import tflite_runtime.interpreter as tflite
import platform
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
from AgeGender.AgeGender_func import age_gender
import sys
from operator import add,sub
import multiprocessing
from Subfunctions import utilities
##Parameters
# NOTE: best settings with acceleration of 90 deg/s² and speedj=40 deg/s
global args
args=False
target_circle_size=60 #in pixels
max_speed=30 #degrees per second
min_speed=2 #degrees per second
action_time_interval=0.1 #for twist commands
beer_time=3 #seconds that robot waits until he hands over the beer
speedj=40 #deg / second
speedl=0.20 #meters / second
testing=False
no_high_five=False #False = High Five enabled

beer_pass_dist=[0,-0.35,0.4,0,0,0] #relative path to pass the beer, a too far reach could be instable
idle_pos=[87.58, 78.68, 353.68, 216.49, 4.0, 336.91, 89.14]
#pre_transport_pos=[233.73, 265.96, 71.5, 212.18, 349.19, 70.11, 8.31] #safe position from where robot can move anywhere
pre_transport_pos=[235.44, 318.67, 71.07, 225.1, 351.47, 53.77, 58.46]
#transport_pos=[257.64, 246.34, 31.6, 218.63, 214.63, 98.69, 129.78] #position with high chance for collisions
transport_pos=[258.69, 241.24, 28.47, 220.11, 210.63, 100.69, 129.75]
high_five_pos=[93.15, 354.41, 355.16, 293.65, 1.47, 71.49, 96.16]
##

#
# #shared variables (between all processes)
# target_middle_shared=multiprocessing.Array("i",[0,0])
# delta_shared=multiprocessing.Array("i",[0,0])
# success_flag_shared=multiprocessing.Value("i",0)
# vision_middle_shared=multiprocessing.Array("i",[0,0])
# h_shared=multiprocessing.Value("i",0)
# w_shared=multiprocessing.Value("i",0)
# elapsed_time=multiprocessing.Value("d",0.0)
# e = multiprocessing.Event() #for movement processes
# e_cam=multiprocessing.Event() #for camera process


def cam(cap_id,e,target_middle_shared,delta_shared,success_flag_shared,vision_middle_shared,h_shared,w_shared,elapsed_time,beer_time):
	global args
	if not args:
		args = utilities.parseConnectionArguments()

	# Create connection to the device and get the router
	with utilities.DeviceConnection.createTcpConnection(args) as router:
		#Start EdgeTPU
		default_model_dir = './models'
		#default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
		default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
		default_labels = 'coco_labels.txt'
		default_threshold = 0.1

		labels = load_labels(os.path.join(default_model_dir,default_labels))
		interpreter = make_interpreter(os.path.join(default_model_dir,default_model))
		interpreter.allocate_tensors()
		cap = WebcamVideoStream(cap_id).start() #multithreading Video Capture
		#Window Settings
		window_name="Robot_Camera"
		size_wh=(1920-300,1080)
		location_xy=(0,0)
		cv2.namedWindow(window_name,
		            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
		cv2.resizeWindow(window_name, *size_wh)
		# cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
		cv2.moveWindow(window_name, *location_xy)
		# cv2.imshow(window_name, image)

		success_flag=False
		t0=time.time()
		auto_focus_time=7
		device_manager = DeviceManagerClient(router)
		vision_config = VisionConfigClient(router)
		vision_device_id = vision_action.example_vision_get_device_id(device_manager)
		#Disable Auto-Focus
		vision_action.autofocus_action(vision_config, vision_device_id,action_id=1)
		while True:
			frame = cap.read()
			(h, w) = frame.shape[:2]
			vision_middle=(int(w/2),int(h/2)-150)
			cv2.circle(frame,vision_middle,target_circle_size*2,(255,0,255),2)
			#Detect the object and get the target middle
			cv2_im = frame
			cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
			pil_im = Image.fromarray(cv2_im_rgb)
			scale = detect.set_input(interpreter,pil_im.size,lambda size: pil_im.resize(size, Image.ANTIALIAS))
			interpreter.invoke()
			objs = detect.get_output(interpreter, default_threshold, scale)
			cv2_im = append_objs_to_img(cv2_im, objs, labels)
			if objs:
				#find the biggest bounding box
				area=0
				for i0 in range(len(objs)):
					x0, y0, x1, y1 = list(objs[i0].bbox)
					area_temp=(x1-x0)*(y1-y0)
					if area_temp>area:
						area=area_temp
						biggest_area_index=i0

				x0, y0, x1, y1 = list(objs[biggest_area_index].bbox)
				target_middle=(int(x0+(x1-x0)/2),int(y0+(y1-y0)/2))
				success_flag=True
				cv2.circle(cv2_im,target_middle,int(target_circle_size/2),(0, 255, 0),2)
			else:
				success_flag=False


			if success_flag==False:
				#human/face was not detected
				target_middle=vision_middle
				if time.time()-t0>auto_focus_time:
					try: #sometimes when stream corrupted because of Schleifring, auotfocus leads to error
						vision_action.autofocus_action(vision_config, vision_device_id,action_id=6,focus_point=target_middle_shared[:])
					except:
						None
					t0=time.time()
			#draw the delta
			delta=[0,0]
			delta[0]=target_middle[0]-vision_middle[0]
			delta[1]=target_middle[1]-vision_middle[1]
			cv2.line(cv2_im,vision_middle,(vision_middle[0]+delta[0],vision_middle[1]+delta[1]),(0,255,0),1)
			#what needs to be given to the process:
			target_middle_shared[0]=target_middle[0]
			target_middle_shared[1]=target_middle[1]
			delta_shared[0]=delta[0]
			delta_shared[1]=delta[1]
			success_flag_shared.value=success_flag
			vision_middle_shared[0]=vision_middle[0]
			vision_middle_shared[1]=vision_middle[1]
			h_shared.value=h
			w_shared.value=w
			if elapsed_time.value>1:
				cv2_im=functions_.draw_loading_circle(
					img=cv2_im, radius=target_circle_size, center=vision_middle,
					elapsed_time=elapsed_time.value-1, end_time=beer_time-1)
			if elapsed_time.value>beer_time:
				#write text
				font = cv2.FONT_HERSHEY_SIMPLEX
				org = (vision_middle[0]-300,vision_middle[1]-100)
				fontScale = 2
				color = (255, 255, 255)
				thickness = 3
				cv2_im = cv2.putText(cv2_im, 'Have a beer, buddy', org, font,
				   fontScale, color, thickness, cv2.LINE_AA)
			#Call to high five
			if elapsed_time.value==-1:
			   #write text
			   font = cv2.FONT_HERSHEY_SIMPLEX
			   org = (vision_middle[0]-550,vision_middle[1])
			   fontScale = 4
			   color = (255, 255, 255)
			   thickness = 4
			   cv2_im = cv2.putText(cv2_im, '    High Five', org, font,
				  fontScale, color, thickness, cv2.LINE_AA)

			if elapsed_time.value==-2:
			   #write text
			   font = cv2.FONT_HERSHEY_SIMPLEX
			   org = (vision_middle[0]-550,vision_middle[1])
			   fontScale = 4
			   color = (255, 255, 255)
			   thickness = 4
			   cv2_im = cv2.putText(cv2_im, '   Good Night', org, font,
				  fontScale, color, thickness, cv2.LINE_AA)

			cv2.imshow(window_name, frame)
			if (cv2.waitKey(1) & 0xFF == ord('q')) or e.is_set():
				break
		cap.stop()
		cv2.destroyWindow(str(cap_id))
		return
def start_cam():
	e.clear()
	p = multiprocessing.Process(target=cam, args=('rtsp://192.168.1.10/color',e_cam,target_middle_shared,delta_shared,success_flag_shared,vision_middle_shared,h_shared,w_shared,elapsed_time,beer_time))
	p.start()

def wave():
	elapsed_time.value=0
	global e
	global args
	if not args:
		args = utilities.parseConnectionArguments()
	with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
		# Create required services
		base = BaseClient(router)
		base_cyclic = BaseCyclicClient(router)
		device_manager = DeviceManagerClient(router)
		#Activate the continuous auto-focus of the camera
		vision_config = VisionConfigClient(router)
		vision_device_id = vision_action.example_vision_get_device_id(device_manager)
		# if vision_device_id != 0:
		# 	vision_action.autofocus_action(vision_config, vision_device_id,action_id=2)

		# check if robot is not in transport position, if so, move safely
		if functions_.pose_comparison(transport_pos,base_cyclic,"joint")==True: #Robot in transport position?
			highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=pre_transport_pos,speed=speedj) #pre transport safety pose
		#Wave Sequence

		wave_speed=70 #degrees/sec
		# wave_start_pose=[180, 0, 0, 0, 90, 90, 270] #joint4
		# wave_left_pose=list(map(add,wave_start_pose,[0,0,0,30,0,0,-30])) #joint4
		# wave_start_pose=[180, 0, 0, 0, 0, 0, 0] #joint5
		# wave_left_pose=list(map(add,wave_start_pose,[0,0,0,0,0,40,0])) #joint4
		# wave_start_pose=[182.49, 36.45, 0.78, 269.14, 2.33, 56.02, 0.11] #joint5
		# wave_left_pose=list(map(add,wave_start_pose,[0,0,0,0,0,40,0])) #joint4
		wave_start_pose=[182.9, 320.18, 177.58, 260.17, 1.06, 65.48, 180.64] #joint5
		wave_left_pose=list(map(add,wave_start_pose,[0,0,0,0,0,40,0])) #joint4

		# highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=wave_start_pose,speed=speedj)
		# sys.exit()
		highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=wave_left_pose,speed=speedj) #okay
		#time.sleep(1)
		highlevel_movements.sinus_move(e,v_max=wave_speed,degrees_move=60,joint_id=5,direction=1,base=base,base_cyclic=base_cyclic)
		time.sleep(0.4) #must be because else meassurements are wrong
		highlevel_movements.sinus_move(e,v_max=wave_speed,degrees_move=60,joint_id=5,direction=-1,base=base,base_cyclic=base_cyclic)
		time.sleep(0.4) #must be because else meassurements are wrong
		highlevel_movements.sinus_move(e,v_max=wave_speed,degrees_move=60,joint_id=5,direction=1,base=base,base_cyclic=base_cyclic)
		time.sleep(0.4) #must be because else meassurements are wrong
		highlevel_movements.sinus_move(e,v_max=wave_speed,degrees_move=60,joint_id=5,direction=-1,base=base,base_cyclic=base_cyclic)
		# time.sleep(0.4)
		# highlevel_movements.sinus_move(e,v_max=wave_speed,degrees_move=60,joint_id=5,direction=1,base=base,base_cyclic=base_cyclic)
		# time.sleep(0.4) #must be because else meassurements are wrong
		# highlevel_movements.sinus_move(e,v_max=wave_speed,degrees_move=60,joint_id=5,direction=-1,base=base,base_cyclic=base_cyclic)
		# time.sleep(0.4) #must be because else meassurements are wrong
		# highlevel_movements.sinus_move(e,v_max=wave_speed,degrees_move=60,joint_id=5,direction=1,base=base,base_cyclic=base_cyclic)
		# time.sleep(0.4) #must be because else meassurements are wrong
		# highlevel_movements.sinus_move(e,v_max=wave_speed,degrees_move=60,joint_id=5,direction=-1,base=base,base_cyclic=base_cyclic)
		elapsed_time.value=-2
		time.sleep(0.4)
		highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=pre_transport_pos,speed=speedj) #pre transport safety pose
		highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=transport_pos,speed=speedj) #transport pose

def start_wave_process():
	global e
	stop_movements()
	e.clear()
	p2 = multiprocessing.Process(target=wave)
	p2.start()
def look_into_crowd():
	global args
	global e
	if not args:
		args = utilities.parseConnectionArguments()
	with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
		# Create required services
		base = BaseClient(router)
		base_cyclic = BaseCyclicClient(router)
		device_manager = DeviceManagerClient(router)
		#Activate the continuous auto-focus of the camera
		vision_config = VisionConfigClient(router)
		vision_device_id = vision_action.example_vision_get_device_id(device_manager)
		# check if robot is not in transport position, if so, move safely
		if functions_.pose_comparison(transport_pos,base_cyclic,"joint")==True: #Robot in transport position?
			highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=pre_transport_pos,speed=speedj) #pre transport safety pose
		#INSERT WAYPOINTS HERE
	return

def start_look_crowd_process():
	global e
	stop_movements()
	e.clear()
	p3 = multiprocessing.Process(target=look_into_crowd)
	p3.start()

def transport_position():
	global args
	global e
	if not args:
		args = utilities.parseConnectionArguments()
	with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
		# Create required services
		base = BaseClient(router)
		base_cyclic = BaseCyclicClient(router)
		#Moving back to init Pose
		if functions_.pose_comparison(transport_pos,base_cyclic,"joint")==False: #Robot already in position?
			success=True
			success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=pre_transport_pos,speed=speedj) #pre transport safety pose
			success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=transport_pos ,speed=speedj) #transport pose

def start_process_transport_position():
	global e
	stop_movements()
	e.clear()
	p5 = multiprocessing.Process(target=transport_position)
	p5.start()

def idle_upright_pos():
	#move into idle upright joint_position
	global args
	global e
	if not args:
		args = utilities.parseConnectionArguments()
	with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
		# Create required services
		base = BaseClient(router)
		base_cyclic = BaseCyclicClient(router)
		if functions_.pose_comparison(transport_pos,base_cyclic,"joint")==True: #Robot in Transport position?
			success=True
			success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=pre_transport_pos,speed=speedj) #pre transport safety pose
			# success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=transport_pos ,speed=speedj) #transport pose
		success=True
		success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=idle_pos,speed=speedj)

def start_process_idle_upright_pos():
	global e
	stop_movements()
	e.clear()
	p4 = multiprocessing.Process(target=idle_upright_pos)
	p4.start()

def read_base_cyclic():
	global args
	global e
	if not args:
		args = utilities.parseConnectionArguments()
	with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
		# Create required services
		base = BaseClient(router)
		base_cyclic = BaseCyclicClient(router)
		tlog=time.time()
		now = datetime.datetime.now()
		#print (now.strftime("%Y-%m-%d %H:%M:%S"))
		timestamp=now.strftime("%Y-%m-%d-%H-%M-%S")
		f = open("./Subfunctions/log_recordings/rec_logging_%s.txt"%timestamp, "w")
		while e.is_set()==False:
			if time.time()-tlog>0.1:
				log_string=""
				log_string+="tool pose "
				log_string+=str(highlevel_movements.get_tcp_pose(base_cyclic))
				log_string+=" at joints "
				log_string+=str(highlevel_movements.get_joint_angles(base_cyclic))
				log_string+="\n"
				f.write(log_string)
				print(log_string)
				tlog=time.time()

def start_read_base_cyclic_process():
	e.clear()
	p6 = multiprocessing.Process(target=read_base_cyclic)
	p6.start()

def stop_movements():
	e.set() #kill movement processes
	time.sleep(0.2)
	global args
	if not args:
		args = utilities.parseConnectionArguments()
	with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
		# Create required services
		base = BaseClient(router)
		base.Stop()
def stop_camera():
	e_cam.set() #kill the camera process

def high_five(base_angle=93.54):
	global args
	global e
	if not args:
		args = utilities.parseConnectionArguments()
	with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
		# Create required services
		base = BaseClient(router)
		base_cyclic = BaseCyclicClient(router)
		temp_speed=5
		high_five_pos[0]=base_angle
		if functions_.pose_comparison(transport_pos,base_cyclic,"joint")==True: #Robot in Transport position?
			success=True
			success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=pre_transport_pos,speed=speedj) #pre transport safety pose
			# success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=transport_pos ,speed=speedj) #transport pose
			temp_speed=10
		success=True
		success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=high_five_pos,speed=speedj)
		time.sleep(1)
		elapsed_time.value=-1
		torque_tool=functions_.TorqueTool()
		torque_tool.tap_toggle(base_cyclic,torque_threshold=8)
		time.sleep(0.1)
		success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=[base_angle, 352.44, 355.16, 329.09, 1.48, 38.79, 96.13],speed=speedj)
		success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=high_five_pos,speed=speedj)
		elapsed_time.value=0
		# success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=idle_pos,speed=speedj)
def start_high_five_process():

	global e
	stop_movements()
	e.clear()
	p7 = multiprocessing.Process(target=high_five)
	p7.start()

def main():
	# Create connection to the device and get the router
	global args
	global e
	if not args:
		args = utilities.parseConnectionArguments()
	with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
		# Create required services
		base = BaseClient(router)
		base_cyclic = BaseCyclicClient(router)
		device_manager = DeviceManagerClient(router)
		#Activate the continuous auto-focus of the camera
		vision_config = VisionConfigClient(router)
		vision_device_id = vision_action.example_vision_get_device_id(device_manager)
		# if vision_device_id != 0:
		# 	vision_action.autofocus_action(vision_config, vision_device_id,action_id=2)
		# Initial Movements
		success = True
		success &= highlevel_movements.send_gripper_command(base,value=0) #open gripper
		if functions_.pose_comparison(transport_pos,base_cyclic,"joint")==True: #Robot already in position?
			success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=pre_transport_pos,speed=speedj) #pre transport pose
			temp_speed=3
		else:
			temp_speed=10 #robot further away so give more time
		success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=[231.4, 266.21, 67.14, 216.41, 346.65, 114.31, 10.59],speed=speedj) #snake
		time.sleep(0.4)
		success &= highlevel_movements.move_to_waypoint_linear(e,base, base_cyclic,[0.02, -0.511, 0.184, 90.689, 3.832, 4.054],speed=3) #pre bottle
		success &= highlevel_movements.move_to_waypoint_linear(e,base, base_cyclic,[0.023, -0.539, 0.184, 90.687, 3.828, 4.049],speed=3) #bottle
		success &= highlevel_movements.send_gripper_command(base,value=0.8)
		#time.sleep(2)
		success &= highlevel_movements.move_to_waypoint_linear(e,base, base_cyclic,[0.032, -0.553, 0.335, 90.691, 3.831, 4.048],speed=3) #move bottle up
		success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=[270.94, 42.99, 348.31, 305.15, 0.84, 286.28, 91.49],speed=speedj) #final
		time.sleep(0.5)
		#INSERT Focus on target_pose
		#search for a face and interrupt when found
		success &= highlevel_movements.example_send_joint_speeds(e,base,speeds=[12, 0, 0, 0, 0, 0, 0],timer=0) #robot rotates until face found
		while success_flag_shared.value==False and e.is_set()==False: #success_flag_shared gets True when face was detected
			time.sleep(0.05)
		base.Stop()
		#Follow Face
		t0=time.time()
		token=False # plays a role when beer has to be handed over after target was in lock for some time
		target_reached_flag=False
		beer_mode=False #hand over beer
		print("searching for person")
		while e.is_set()==False:
			if time.time()-t0>action_time_interval and success_flag_shared.value==True and beer_mode==False :
				#calculate the pose increment and rotate robot into target
				target_reached_flag=functions_.rotate_to_target(
					delta_shared[:],w_shared.value,h_shared.value,max_speed,min_speed,
					target_circle_size,base,base_cyclic)
				t0=time.time()
			elif time.time()-t0>action_time_interval and success_flag_shared.value==False and beer_mode==False:
				#no target in sight
				base.Stop()
				# if no target was detected, search with base rotation
				if time.time()-t0>3:
					#search for a face and interrupt when found
					highlevel_movements.example_send_joint_speeds(e,base,speeds=[12, 0, 0, 0, 0, 0, 0],timer=0) #robot rotates until face found
					while success_flag_shared.value==False and e.is_set()==False: #success_flag_shared gets True when face was detected
						time.sleep(0.05)
					base.Stop()
					t0=time.time()


	        #after robot is Xs in target, hand over beer
			if (target_reached_flag==True and token==True and success_flag_shared.value==True) or beer_mode==True:
				elapsed_time.value=time.time()-beer_timer
				if elapsed_time.value>beer_time:
					if beer_mode==False:
						reach_beer_twist_timer=time.time()
					beer_mode=True
					#the target was long enough in scope
					tool_twist_duration=6
					if time.time()-reach_beer_twist_timer<tool_twist_duration:
						highlevel_movements.tool_twist_time(e,base,tool_twist_duration,pose_distance=beer_pass_dist)
					else:
						base.Stop()
						time.sleep(2) #swing out
						#wait until force applied to robot
						torque_tool=functions_.TorqueTool()
						torque_tool.tap_toggle(base_cyclic,torque_threshold=5)
						#open gripper
						highlevel_movements.send_gripper_command(base,value=0.0)
						time.sleep(2)
						#move into idle upright joint_position
						#success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=[87.58, 78.68, 353.68, 216.49, 4.0, 336.91, 89.14],speed=speedj)
						print("beer was passed")
						break

			elif target_reached_flag==True and success_flag_shared.value==True:
				beer_timer=time.time()
				token=True
			else:
				token=False
				elapsed_time.value=0
		if e.is_set()==True or no_high_five==True:
			return
		else:
			# move robot up so he is on face eight again
			rev_beer_pass_dist = [i * (-1) for i in beer_pass_dist]
			tool_twist_duration=6
			# highlevel_movements.tool_twist_time(e,base,tool_twist_duration,pose_distance=beer_pass_dist)
			highlevel_movements.tool_twist_time(e,base,tool_twist_duration,pose_distance=rev_beer_pass_dist)
			# print("Done")
			#sys.exit()
			# follow a little then high five to person
			t0=time.time()
			t0_wave=time.time()
			time_to_wave=False
			while e.is_set()==False:
				if time.time()-t0>action_time_interval and success_flag_shared.value==True and time_to_wave==False:
					target_reached_flag=functions_.rotate_to_target(
						delta_shared[:],target_middle_shared[:],vision_middle_shared[:],w_shared.value,h_shared.value,max_speed,min_speed,
						target_circle_size,base,base_cyclic)
					t0=time.time()
				elif time.time()-t0>action_time_interval and success_flag_shared.value==False and time_to_wave==False:
					base.Stop()
				if time.time()-t0_wave>5:
					time_to_wave=True
					base.Stop()
					time.sleep(0.5)
					break
			if time.time()-t0_wave>5:
				elapsed_time.value=-1
				print("time to high five!")
				#read the joint angles to get the base angle
				pose=highlevel_movements.get_joint_angles(base_cyclic)
				base_angle=pose[0]
				#enter high five position with base rotation
				# high_five(base_angle=base_angle)
				high_five_pos[0]=base_angle
				success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=high_five_pos,speed=speedj)
				torque_tool=functions_.TorqueTool()
				torque_tool.tap_toggle(base_cyclic,torque_threshold=12)
				time.sleep(0.1)
				success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=[base_angle, 352.44, 355.16, 329.09, 1.48, 38.79, 96.13],speed=speedj)
				success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=high_five_pos,speed=speedj)
				elapsed_time.value=-2
				# transport_position()
				success=True
				success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=pre_transport_pos,speed=speedj) #pre transport safety pose
				success &= highlevel_movements.angular_action_movement(e,base,base_cyclic,joint_positions=transport_pos ,speed=speedj) #transport pose



		#process finished
		base.Stop()

def start_robot_main():
	global e
	stop_movements()
	e.clear()
	p1 = multiprocessing.Process(target=main)
	p1.start()
if __name__ == '__main__':
	print("please execute GUI.py")
