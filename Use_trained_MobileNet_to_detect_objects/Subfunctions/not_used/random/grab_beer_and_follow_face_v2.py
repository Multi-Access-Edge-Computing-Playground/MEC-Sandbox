# Lint as: python3
#Has one short process with camer stream that gets aborded
#CONTINUE
"""Statt Tool Base twist machen?"""

"""Using TF Lite to detect objects from camera."""
from imutils.video import WebcamVideoStream
from Subfunctions import functions_
from Subfunctions import highlevel_movements
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
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
import sys
from operator import add,sub
import multiprocessing
from Subfunctions import utilities
##Parameters
enable_stream=False
target_circle_size=40 #in pixels
max_speed=16 #degrees per second
min_speed=3 #degrees per second
action_time_interval=0.1
beer_time=4 #seconds that robot waits until he hands over the beer
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

def cam(cap_id,e):
	args = utilities.parseConnectionArguments()
	# Create connection to the device and get the router
	with utilities.DeviceConnection.createTcpConnection(args) as router:
	    cap = WebcamVideoStream(cap_id).start()
	    while True:
	        frame = cap.read()
	        if e.is_set():
	            break
	        cv2.imshow("robot camera", frame)
	        if cv2.waitKey(1) & 0xFF == ord('q'):
	            break
	    cap.stop()
	    #cv2.destroyWindow(str(cap_id))
	    return

if __name__ == '__main__':
	# Parse arguments
	args = utilities.parseConnectionArguments()
	# Create connection to the device and get the router
	e = multiprocessing.Event()
	p = multiprocessing.Process(target=cam, args=('rtsp://192.168.1.10/color',e))
	p.start()
	with utilities.DeviceConnection.createTcpConnection(args) as router:
		#open camera
		#cap = cv2.VideoCapture('rtsp://192.168.1.10/color')

		#start a video stream process

		# Create required services
		base = BaseClient(router)
		base_cyclic = BaseCyclicClient(router)

		#Start EdgeTPU
		default_model_dir = './models'
		#default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
		default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
		default_labels = 'coco_labels.txt'
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument('-m', '--model', help='File path of .tflite file.', default=os.path.join(default_model_dir,default_model))
		parser.add_argument('-l', '--labels', help='File path of labels file.',default=os.path.join(default_model_dir,default_labels))
		parser.add_argument('-t', '--threshold', type=float, default=0.1, help='Score threshold for detected objects.')
		#parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
		args = parser.parse_args()
		labels = load_labels(args.labels) if args.labels else {}
		interpreter = make_interpreter(args.model)
		interpreter.allocate_tensors()

		# Initial Movements
		success = True
		#Move into transport position (home)
		home_pose=[184.19,291.7,171.7,213.3,181.8,45.8,266.6]
		#success &= highlevel_movements.angular_action_movement(base,joint_positions=home_pose,speed=5)
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[-0.06,-0.096,0.8,-90.823,171.8,113.73],speed=0.1)
		#sys.exit()
		# home_pose=[-0.154,0,0.34,-55.9,173.3,86.6]
		# success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,waypoint=home_pose,speed=0.1)
		#Exectue Sequence
		#success &= highlevel_movements.call_kinova_web_sequence(base, base_cyclic,seq_name="grab_beer_v3")
		#sys.exit()
		#Execute Sequence V2
		"""
		speedj=5
		speedl=0.1
		success &= highlevel_movements.angular_action_movement(base,joint_positions=[89.695, 336.743, 176.642, 232.288, 180.629, 70.981, 272.165],speed=speedj)
		success &= highlevel_movements.angular_action_movement(base,joint_positions=[91.583, 23.663, 174.547, 265.846, 180.949, 35.446, 272.106],speed=speedj)
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.01796681061387062, -0.6095998287200928, 0.2607220709323883, -89.98168182373047, -176.41896057128906, 178.88327026367188],speed=speedl)
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.015759950503706932, -0.6483935713768005, 0.2543827295303345, -91.34257507324219, 178.12986755371094, 178.2921905517578],speed=speedl)
		# GRIPPER_ACTION
		highlevel_movements.send_gripper_command(base,value=0.7)
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.016505524516105652, -0.65036940574646, 0.38649141788482666, -92.8912353515625, 178.2748565673828, 179.34559631347656],speed=speedl)
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[0.07478067278862, -0.1246325895190239, 0.8614432215690613, 90.16726684570312, 9.310687065124512, 5.854083061218262],speed=speedl)
		success &= highlevel_movements.angular_action_movement(base,joint_positions=[179.62, 307.879, 172.652, 289.174, 180.408, 69.243, 272.359],speed=speedj)

		#move to surveillance position
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[-0.06,-0.096,0.8,-90.823,171.8,113.73],speed=0.1)
		"""

		time.sleep(5)
		e.set() #kill the camera process
		p.join()
		cap = WebcamVideoStream(src='rtsp://192.168.1.10/color').start()
		t0=time.time()
		token=False
		target_reached_flag=False
		while True:
			#ret, frame = cap.read()
			frame = cap.read()
			# if not ret:
			# 	break
			(h, w) = frame.shape[:2]
			vision_middle=(int(w/2),int(h/2))
			cv2.circle(frame,vision_middle,target_circle_size*2,(255,0,255),2)
			#Detect the object and get the target middle
			cv2_im = frame
			cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
			pil_im = Image.fromarray(cv2_im_rgb)
			#common.set_input(interpreter, pil_im)
			scale = detect.set_input(interpreter,pil_im.size,lambda size: pil_im.resize(size, Image.ANTIALIAS))
			interpreter.invoke()
			#print(scale)
			objs = detect.get_output(interpreter, args.threshold, scale)
			#print(objs)
			#draw_objects(ImageDraw.Draw(pil_im), objs, labels)
			cv2_im = append_objs_to_img(cv2_im, objs, labels)
			#the robot will only target the first object (human/face)
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
				base.Stop()
				target_middle=vision_middle
			#draw the delta
			delta=[0,0]
			delta[0]=target_middle[0]-vision_middle[0]
			delta[1]=target_middle[1]-vision_middle[1]
			cv2.line(cv2_im,vision_middle,(vision_middle[0]+delta[0],vision_middle[1]+delta[1]),(0,255,0),1)

			if time.time()-t0>action_time_interval and success_flag==True:
				#calculate the pose increment and move robot inot target
				target_reached_flag=functions_.rotate_to_target(
					delta,target_middle,vision_middle,w,h,max_speed,min_speed,
					target_circle_size,base)
				t0=time.time()

	        #after robot is 3s in target, hand over beer
			if target_reached_flag==True and token==True and success_flag==True:
				elapsed_time=time.time()-beer_timer
				if elapsed_time>1:
					cv2_im=functions_.draw_loading_circle(
						img=cv2_im, radius=target_circle_size, center=vision_middle,
						elapsed_time=elapsed_time-1, end_time=beer_time-1)
				if elapsed_time>beer_time:
					#the target was long enough in scope
					#write text
					font = cv2.FONT_HERSHEY_SIMPLEX
					org = (vision_middle[0]-300,vision_middle[1]-100)
					fontScale = 2
					color = (255, 255, 255)
					thickness = 3
					cv2_im = cv2.putText(cv2_im, 'Have a beer, buddy!', org, font,
			           fontScale, color, thickness, cv2.LINE_AA)
					#do pose trans move
					# startpose = highlevel_movements.get_tcp_pose(base_cyclic)
					# beer_pose = highlevel_movements.posetrans(startpose,translation=[0,-0.4,1],rotation=[0,0,0])
					# highlevel_movements.move_to_waypoint_linear(base,base_cyclic,beer_pose,0.1)
			elif target_reached_flag==True and success_flag==True:
				beer_timer=time.time()
				token=True
			else:
				token=False
			cv2.imshow("robot camera", cv2_im)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				base.Stop()
				break

		cap.stop()
		cv2.destroyAllWindows()
