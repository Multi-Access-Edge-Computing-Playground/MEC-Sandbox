# Lint as: python3
#Has one short process with camer stream that gets aborded
#CONTINUE

# Für NUC:
# schalte zwischen Python3.5 und 3.7 mit  sudo update-alternatives --config python3
# um
"""Statt Tool Base twist machen?"""

"""Using TF Lite to detect objects from camera."""
from imutils.video import WebcamVideoStream
from Subfunctions import functions_
from Subfunctions import highlevel_movements
from Subfunctions import vision_action
import argparse
import time
import os
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
enable_stream=False
target_circle_size=60 #in pixels
max_speed=30 #degrees per second
min_speed=2 #degrees per second
action_time_interval=0.1 #for twist commands
beer_time=3 #seconds that robot waits until he hands over the beer
speedj=10 #seconds NOTE: sadly the synchronized angular movement does not work... only when time is defined
speedl=0.20
testing=False
##

EDGETPU_SHARED_LIB = {
	'Linux': 'libedgetpu.so.1',
	'Darwin': 'libedgetpu.1.dylib',
	'Windows': 'edgetpu.dll'
		}[platform.system()]


def load_labels(path, encoding='utf-8'):
	"""Loads labels from file (with or without index numbers).

	Args:
		path: path to label file.
		encoding: label file encoding.
	Returns:
		Dictionary mapping indices to labels.
	"""
	with open(path, 'r', encoding=encoding) as f:
		lines = f.readlines()
		if not lines:
			return {}

		if lines[0].split(' ', maxsplit=1)[0].isdigit():
			pairs = [line.split(' ', maxsplit=1) for line in lines]
			return {int(index): label.strip() for index, label in pairs}
		else:
			return {index: line.strip() for index, line in enumerate(lines)}

def make_interpreter(model_file):
	model_file, *device = model_file.split('@')
	return tflite.Interpreter(
			model_path=model_file,
			experimental_delegates=[tflite.load_delegate(
								EDGETPU_SHARED_LIB,{'device': device[0]} if device else {}
								)]
						)

def draw_objects(draw, objs, labels):
	"""Draws the bounding box and label for each object."""
	for obj in objs:
		bbox = obj.bbox
		draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],outline='red')
		draw.text((bbox.xmin + 10, bbox.ymin + 10),	'%s\n%.2f' % (labels[obj.label_id], obj.score),fill='red')

def append_objs_to_img(cv2_im, objs, labels):
	height, width, channels = cv2_im.shape
	#print("height, width, channels: ",height, width, channels)
	for obj in objs:
		x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
		#print("Part1 x0, y0, x1, y1: ",x0, y0, x1, y1)
		#x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
		#print("x0, y0, x1, y1: ",x0, y0, x1, y1)
		percent = int(100 * obj.score)
		label = '{}% {}'.format(percent, labels[obj.label_id])
		#print("label: ",label)
		cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
		cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
	return cv2_im

def cam(cap_id,e,target_middle_shared,delta_shared,success_flag_shared,vision_middle_shared,h_shared,w_shared,elapsed_time,beer_time):
	#args = utilities.parseConnectionArguments()

	# Create connection to the device and get the router
	with utilities.DeviceConnection.createTcpConnection(args) as router:
		#Start EdgeTPU
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
		default_model_dir = './models'
		#default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
		default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
		default_labels = 'coco_labels.txt'
		default_threshold = 0.1
		# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		# parser.add_argument('-m', '--model', help='File path of .tflite file.', default=os.path.join(default_model_dir,default_model))
		# parser.add_argument('-l', '--labels', help='File path of labels file.',default=os.path.join(default_model_dir,default_labels))
		# parser.add_argument('-t', '--threshold', type=float, default=default_threshold, help='Score threshold for detected objects.')
		# #parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
		# args = parser.parse_args()

		labels = load_labels(os.path.join(default_model_dir,default_labels))
		interpreter = make_interpreter(os.path.join(default_model_dir,default_model))
		interpreter.allocate_tensors()
		cap = WebcamVideoStream(cap_id).start()
		old_face_position=0
		face_gone_timer=time.time()
		success_flag=False
		while True:
			frame = cap.read()
			(h, w) = frame.shape[:2]
			vision_middle=(int(w/2),int(h/2)-150)
			cv2.circle(frame,vision_middle,target_circle_size*2,(255,0,255),2)
			#Detect the object and get the target middle
			cv2_im = frame
			cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
			pil_im = Image.fromarray(cv2_im_rgb)
			#common.set_input(interpreter, pil_im)
			scale = detect.set_input(interpreter,pil_im.size,lambda size: pil_im.resize(size, Image.ANTIALIAS))
			interpreter.invoke()
			#print(scale)
			objs = detect.get_output(interpreter, default_threshold, scale)
			#print(objs)
			#draw_objects(ImageDraw.Draw(pil_im), objs, labels)
			cv2_im = append_objs_to_img(cv2_im, objs, labels)
			if testing == True:
				#the robot will only target the first object (human/face)
				faces_list=[]
				if objs:
					target_middle=0
					for face_id in range(len(objs)):
						x0, y0, x1, y1 = list(objs[face_id].bbox)
						target_middle_temp=(int(x0+(x1-x0)/2),int(y0+(y1-y0)/2))
						cv2.circle(cv2_im,target_middle_temp,int(target_circle_size/2),(0, 255, 0),2)
						faces_list.append([face_id,target_middle_temp])
						# print("old_face_position: ",old_face_position)
						# print("faces_list[face_id][1]: ",faces_list[face_id][1])

						if old_face_position==0: #never seen a face before, first assignment:
							old_face_position=target_middle_temp
							target_middle=target_middle_temp
							#success_flag=True
						elif ((old_face_position[0]-(x1-x0)<faces_list[face_id][1][0]<old_face_position[0]+(x1-x0)) and
							(old_face_position[1]-(x1-x0)<faces_list[face_id][1][1]<old_face_position[1]+(x1-x0))):
							target_middle=(int(x0+(x1-x0)/2),int(y0+(y1-y0)/2))
							old_face_position=target_middle
							success_flag=True
							face_gone_timer=time.time()
					# if time.time()-face_gone_timer>2 and success_flag==False: #if the face left the image, after 2 second set old_face_position=0
					# 	old_face_position=0

					#cv2_im = age_gender(cv2_im,face_boxes=objs)

				else:
					success_flag=False

				if success_flag==False:
					#human/face was not detected
					#base.Stop()
					target_middle=vision_middle
			else:
				if objs:
					x0, y0, x1, y1 = list(objs[0].bbox)
					target_middle=(int(x0+(x1-x0)/2),int(y0+(y1-y0)/2))
					#print("target_middle: ",target_middle)
					success_flag=True
					cv2.circle(cv2_im,target_middle,int(target_circle_size/2),(0, 255, 0),2)
				else:
					success_flag=False

				if success_flag==False:
					#human/face was not detected
					#base.Stop()
					target_middle=vision_middle
			#draw the delta
			delta=[0,0]
			delta[0]=target_middle[0]-vision_middle[0]
			delta[1]=target_middle[1]-vision_middle[1]
			cv2.line(cv2_im,vision_middle,(vision_middle[0]+delta[0],vision_middle[1]+delta[1]),(0,255,0),1)
			#what needs to be given to othe process:
			#target_middle,delta,success_flag
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
				cv2_im = cv2.putText(cv2_im, 'Have a beer, buddy!', org, font,
				   fontScale, color, thickness, cv2.LINE_AA)

			#cv2.imshow("robot camera", frame)
			#send the image to the client
		    result, frame = cv2.imencode('.jpg', frame, encode_param)
		    #data = zlib.compress(pickle.dumps(frame, 0))
		    data = pickle.dumps(frame, 0)
		    size = len(data)
		    client_socket.sendall(struct.pack("L", size) + data)
			# if cv2.waitKey(1) & 0xFF == ord('q') or e.is_set():
			# 	break
		cap.stop()
		cv2.destroyWindow(str(cap_id))
		return


def main():
	target_middle_shared=multiprocessing.Array("i",[0,0])
	delta_shared=multiprocessing.Array("i",[0,0])
	success_flag_shared=multiprocessing.Value("i",0)
	vision_middle_shared=multiprocessing.Array("i",[0,0])
	h_shared=multiprocessing.Value("i",0)
	w_shared=multiprocessing.Value("i",0)
	elapsed_time=multiprocessing.Value("d",0.0)
	args = utilities.parseConnectionArguments()
	# Create connection to the device and get the router
	e = multiprocessing.Event()
	p = multiprocessing.Process(target=cam, args=('rtsp://192.168.1.10/color',e,target_middle_shared,delta_shared,success_flag_shared,vision_middle_shared,h_shared,w_shared,elapsed_time,beer_time))
	p.start()
	with utilities.DeviceConnection.createTcpConnection(args) as router:
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
		#Move into transport position (home)
		home_pose=[184.19,291.7,171.7,213.3,181.8,45.8,266.6]
		#success &= highlevel_movements.angular_action_movement(base,joint_positions=home_pose,speed=5)
		#success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[-0.06,-0.096,0.8,-90.823,171.8,113.73],speed=speedl)


		# Grab the beer
		#Desk Sequence
		# highlevel_movements.send_gripper_command(base,value=0.0)
		# #success &= highlevel_movements.angular_action_movement(base,joint_positions=[89.695, 336.743, 176.642, 232.288, 180.629, 70.981, 272.165],speed=speedj)
		# success &= highlevel_movements.angular_action_movement(base,joint_positions=[91.583, 23.663, 174.547, 265.846, 180.949, 35.446, 272.106],speed=speedj)
		# success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.01796681061387062, -0.6095998287200928, 0.2607220709323883, -89.98168182373047, -176.41896057128906, 178.88327026367188],speed=speedl)
		# success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.015759950503706932, -0.6483935713768005, 0.2543827295303345, -91.34257507324219, 178.12986755371094, 178.2921905517578],speed=speedl)
		# # GRIPPER_ACTION
		# success &= highlevel_movements.send_gripper_command(base,value=0.7)
		# time.sleep(1)
		# success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.016505524516105652, -0.65036940574646, 0.38649141788482666, -92.8912353515625, 178.2748565673828, 179.34559631347656],speed=speedl)
		# success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.07478067278862, -0.1246325895190239, 0.8614432215690613, 90.16726684570312, 9.310687065124512, 5.854083061218262],speed=speedl)
		# # move to surveillance position
		#success &= highlevel_movements.angular_action_movement(base,joint_positions=[179.62, 307.879, 172.652, 289.174, 180.408, 69.243, 272.359],speed=speedj)
		#move to surveillance position
		#success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[-0.06,-0.096,0.8,-90.823,171.8,113.73],speed=0.1)

		#Dog Sequence Desired Pose : [249.37, 239.26, 223.77, 133.81, 217.24, 259.69, 307.95]
		#TODO Startposition [249.37, 239.26, 223.77, 133.81, 217.24, 259.69, 307.95] check
		#TODO Schwenk ins Publikum nach vorne (extra funktion, muss aufgerufen werden) Tkinter
		#TODO Winken nur mit oberen zwei Gelenken Tkinter
		#TODO Theta_y korrigieren
		#TODO Sicht korrigieren (Fokus und ggf. heller/Dunkler)
		# success &= highlevel_movements.angular_action_movement(base,joint_positions=[249.37, 239.26, 223.77, 133.81, 217.24, 259.69, 307.95],speed=10) #Dog Pose
		# sys.exit()
		success &= highlevel_movements.send_gripper_command(base,value=0) #open gripper
		success &= highlevel_movements.angular_action_movement(base,joint_positions=[233.73, 265.96, 71.5, 212.18, 349.19, 70.11, 8.31],speed=speedj) #pre snake pose
		success &= highlevel_movements.angular_action_movement(base,joint_positions=[257.64, 246.34, 31.6, 218.63, 214.63, 98.69, 129.78],speed=3) #desired dog pose
		#success &= highlevel_movements.angular_action_movement(base,joint_positions=[233.73, 265.96, 71.5, 212.18, 349.19, 70.11, 8.31],speed=speedj) #pre snake pose
		success &= highlevel_movements.angular_action_movement(base,joint_positions=[238.78, 264.98, 70.21, 217.99, 349.19, 120.32, 8.3],speed=3) #snake
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.006, -0.536, 0.191, 88.99, 1.239, 7.97],speed=speedl)
		#GRIPPER_ACTION
		success &= highlevel_movements.send_gripper_command(base,value=0.4)
		time.sleep(1.5)
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.008, -0.535, 0.275, 88.988, 1.245, 7.968],speed=speedl) #move a little up
		#success &= highlevel_movements.angular_action_movement(base,joint_positions=[235.7, 344.71, 64.58, 219.35, 49.72, 80.94, 54.84],speed=speedj) # upright1
		#success &= highlevel_movements.angular_action_movement(base,joint_positions=[231.32, 353.1, 32.7, 304.08, 37.1, 321.32, 58.52],speed=speedj) #upright 2
		success &= highlevel_movements.angular_action_movement(base,joint_positions=[270.94, 42.99, 348.31, 305.15, 0.84, 286.28, 91.49],speed=speedj) #final
		#success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.117, -0.058, 0.914, 86.046, 9.61, 6.732],speed=speedl) #final

		#sys.exit()
		#search for a face and interrupt when found
		#vision_action.autofocus_action(vision_config, vision_device_id,action_id=4)

		success &= highlevel_movements.example_send_joint_speeds(base,speeds=[18, 0, 0, 0, 0, 0, 0],timer=20,success_flag_shared=success_flag_shared)
		#sys.exit()

		#Find Face
		t0=time.time()
		token=False
		target_reached_flag=False
		beer_mode=False
		print("searching for person")
		while True:
			if time.time()-t0>action_time_interval and success_flag_shared.value==True and beer_mode==False and p.is_alive():
				#calculate the pose increment and move robot into target
				target_reached_flag=functions_.rotate_to_target(
					delta_shared[:],target_middle_shared[:],vision_middle_shared[:],w_shared.value,h_shared.value,max_speed,min_speed,
					target_circle_size,base,base_cyclic)
				t0=time.time()
			elif time.time()-t0>action_time_interval and success_flag_shared.value==False and beer_mode==False:
				base.Stop()
				# if time.time()-t0>2:
				# 	vision_action.autofocus_action(vision_config, vision_device_id,action_id=6,focus_point=target_middle_shared[:])
			elif p.is_alive()==False:
				base.Stop()
				print("Process aborded")
				sys.exit()

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
						highlevel_movements.tool_twist_time(base,tool_twist_duration,pose_distance=[0,-0.35,0.4,0,0,0])
					else:
						base.Stop()
						time.sleep(2) #swing out
						#wait until force applied to robot
						torque_tool=functions_.TorqueTool()
						torque_tool.tap_toggle(base_cyclic,torque_threshold=5)
						#open gripper
						highlevel_movements.send_gripper_command(base,value=0.0)
						time.sleep(2)

						home_pose=[184.19,291.7,171.7,213.3,181.8,45.8,266.6] #Desk
						#success &= highlevel_movements.angular_action_movement(base,joint_positions=[267.038, 239.612, 177.453, 212.171, 185.765, 56.095, 269.943],speed=speedj)

						highlevel_movements.tool_twist_time(base,tool_twist_duration,pose_distance=[0,-0.35,0.4,0,0,0])

						#Wave Sequence
						wave_start_pose=[180, 0, 0, 0, 82, 90, 270]
						wave_left_pose=list(map(add,wave_start_pose,[0,0,0,30,0,0,-30]))
						wave_time=2 #seconds
						#wave_left_pose=list(map(add,wave_start_pose,[0,-30,0,-30,0,0,0]))
						highlevel_movements.angular_action_movement(base,joint_positions=wave_start_pose,speed=4)
						#while True:
						highlevel_movements.angular_action_movement(base,joint_positions=wave_left_pose,speed=2.5) #okay
						highlevel_movements.angular_action_movement(base,joint_positions=[180, 0, 0, 330, 82, 90, 300],speed=4.5)
						highlevel_movements.angular_action_movement(base,joint_positions=wave_left_pose,speed=4.5)
						highlevel_movements.angular_action_movement(base,joint_positions=[180, 0, 0, 330, 82, 90, 300],speed=4.5)
						highlevel_movements.angular_action_movement(base,joint_positions=wave_start_pose,speed=2.5)
						#sys.exit()

						#Moving back to init Pose
						success &= highlevel_movements.angular_action_movement(base,joint_positions=[233.73, 265.96, 71.5, 212.18, 349.19, 70.11, 8.31],speed=speedj) #pre snake pose
						success &= highlevel_movements.angular_action_movement(base,joint_positions=[257.64, 246.34, 31.6, 218.63, 214.63, 98.69, 129.78],speed=3) #desired dog pose

						#success &= highlevel_movements.angular_action_movement(base,joint_positions=home_pose,speed=speedj)
						e.set() #kill the camera process
						break

			elif target_reached_flag==True and success_flag_shared.value==True:
				beer_timer=time.time()
				token=True
			else:
				token=False
				elapsed_time.value=0
if __name__ == '__main__':
	main()
