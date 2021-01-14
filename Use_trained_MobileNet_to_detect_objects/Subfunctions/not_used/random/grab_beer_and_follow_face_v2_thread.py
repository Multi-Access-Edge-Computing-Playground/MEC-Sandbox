# Lint as: python3

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
import threading
##Parameters
enable_stream=False
target_circle_size=40 #in pixels
max_speed=7 #degrees per second
min_speed=3 #degrees per second
action_time_interval=0.1
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

from Subfunctions import utilities
# Parse arguments
args = utilities.parseConnectionArguments()
# Create connection to the device and get the router
with utilities.DeviceConnection.createTcpConnection(args) as router:
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

	def init_movements(i,base,base_cyclic,return_val):
		# Initial Movements
		success = True
		#Move into transport position (home)
		home_pose=[184.19,291.7,171.7,213.3,181.8,45.8,266.6]
		#success &= highlevel_movements.angular_action_movement(base,joint_positions=home_pose,speed=5)
		#sys.exit()
		# home_pose=[-0.154,0,0.34,-55.9,173.3,86.6]
		# success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,waypoint=home_pose,speed=0.1)
		#Exectue Sequence
		#success &= highlevel_movements.call_kinova_web_sequence(base, base_cyclic,seq_name="Grab_Beer")
		#sys.exit()
		#move to surveillance position
		success &= highlevel_movements.move_to_waypoint_linear(base, base_cyclic,[-0.06,-0.096,0.8,-90.823,171.8,113.73],speed=0.05)
		return_val=success
		

	# manager = multiprocessing.Manager()
	# return_dict = manager.dict()
	# e = multiprocessing.Event()
	# p = multiprocessing.Process(target=init_movements, args=(1,base,base_cyclic,return_dict))
	# p.start()
	return_val=0
	t=threading.Thread(target=init_movements,args=(1,base,base_cyclic,return_val))
	t.start()

	#open camera
	#cap = cv2.VideoCapture('rtsp://192.168.1.10/color')
	cap = WebcamVideoStream(src='rtsp://192.168.1.10/color').start()
	t0=time.time()
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
		cv2.imshow('frame', cv2_im)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			base.Stop()
			break
		if time.time()-t0>action_time_interval:
			#calculate the pose increment to hit the target
			#functions_.rotate_to_target(delta,target_middle,vision_middle,w,h,max_speed,min_speed,target_circle_size,base)
			print(return_val)
			t0=time.time()
	cap.stop()
	cv2.destroyAllWindows()

	# image = Image.open(args.input)
	# scale = detect.set_input(interpreter,
		#												 image.size,
		#												 lambda size: image.resize(size, Image.ANTIALIAS)
		#												 )
		#
	# interpreter.invoke()
	# objs = detect.get_output(interpreter, args.threshold, scale)
	# # for obj in objs:
	# #	 print(labels[obj.label_id])
	# #	 print('	id: ', obj.id)
	# #	 print('	score: ', obj.score)
	# #	 print('	bbox: ', obj.bbox)
		#
	# if args.output:
	#	 image = image.convert('RGB')
	#	 draw_objects(ImageDraw.Draw(image), objs, labels)
	#	 image.save(args.output)
	#	 image.show()
