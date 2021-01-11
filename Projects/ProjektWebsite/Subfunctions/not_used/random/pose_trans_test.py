# Lint as: python3

#TODO von S-Bus zu Python (later)
#TODO Sicherheitsfeatures für Ellbogen und Handgelenk erweitern (later)
#TODO Camera Feed Fehler abfangen (later)
#TODO set global acceleration via API

#TODO smooth waving with sinus
import sys
import math
#remove path to ROS path, so cv2 can work
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from Subfunctions import functions_
from Subfunctions import highlevel_movements
from Subfunctions import vision_action
import CAMERA_Process
import argparse
import time
import datetime
import os
import signal
from PIL import Image
from PIL import ImageDraw
import cv2
from Subfunctions import detect
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
from scipy.spatial.transform import Rotation as R
import numpy as np


from operator import add,sub
import multiprocessing
from Subfunctions import utilities

global args
args=False
enable_stream=False
target_circle_size=60 #in pixels
max_speed=30 #degrees per second
min_speed=2 #degrees per second
action_time_interval=0.1 #for twist commands
beer_time=3 #seconds that robot waits until he hands over the beer
speedj=40 #deg / second
speedl=0.20 #meters / second
testing=False
no_high_five=False #False = High Five enabled
e = multiprocessing.Event() #for movement processes
beer_pass_dist=[0,-0.35,0.4,0,0,0] #relative path to pass the beer, a too far reach could be instable
idle_pos=[87.58, 78.68, 353.68, 216.49, 4.0, 336.91, 89.14]
#pre_transport_pos=[233.73, 265.96, 71.5, 212.18, 349.19, 70.11, 8.31] #safe position from where robot can move anywhere
pre_transport_pos=[235.44, 318.67, 71.07, 225.1, 351.47, 53.77, 58.46]
#transport_pos=[257.64, 246.34, 31.6, 218.63, 214.63, 98.69, 129.78] #position with high chance for collisions
transport_pos=[258.69, 241.24, 28.47, 220.11, 210.63, 100.69, 129.75]
high_five_pos=[93.15, 354.41, 355.16, 293.65, 1.47, 71.49, 96.16]

if not args:
    args = utilities.parseConnectionArguments()
with utilities.DeviceConnection.createTcpConnection(args) as router: #kann das "with" weg? Nein
    # Create required services
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)
    device_manager = DeviceManagerClient(router)

    # highlevel_movements.angular_action_movement(
    #     e,base,base_cyclic,joint_positions=[270.94+90, 42.99, 348.31, 305.15, 0.84, 286.28, 91.49],
    #     speed=6,watch_for_flag=True) #final
    # while True:
    current_pose=highlevel_movements.get_tcp_pose(base_cyclic)
    print("current_pose: ",current_pose)
    joint_angles=highlevel_movements.get_joint_angles(base_cyclic)
    pose_tait_bryan=functions_.forward_kin(joint_angles)
    print("pose_tait_bryan: ",pose_tait_bryan)
    delta_angles=np.array(current_pose[0:6])-np.array(pose_tait_bryan[0:6])
    print("delta_angles: ",delta_angles)
    delta_angles=np.linalg.norm(np.array(current_pose[0:3])-np.array(pose_tait_bryan[0:3]))
    print("Magnitude: ",delta_angles)

    highlevel_movements.move_to_waypoint_linear(e, base, base_cyclic, pose_tait_bryan,speed=0.05)
    # time.sleep(1)
    # current_pose=highlevel_movements.get_tcp_pose(base_cyclic)
    # print("current_pose: ",current_pose)
    # new_pose=CAMERA_Process.posetrans(current_pose,[0,0,0,0,0,0])
    # print("new_pose: ",new_pose)
    # highlevel_movements.move_to_waypoint_linear(e, base, base_cyclic, new_pose,speed=0.1)
    # sys.exit()
    #current_pose[3:6]=[math.radians(num) for num in current_pose[3:6]]
    print("--------------------")
    current_pose=highlevel_movements.get_tcp_pose(base_cyclic)
    print("current_pose: ",current_pose)
    new_pose=functions_.posetrans(current_pose,[0,0,0.05,0,0,0])
    new_pose=[round(num, 3) for num in new_pose]
    print("new_pose:     ",new_pose)
    # delta=np.array(new_pose)-np.array(current_pose)
    # print("delta: ",list(np.array(new_pose)-np.array(current_pose)))
    # new_pose[3:6]=list(np.array(new_pose[3:6])-delta[3:6])
    # print("corrected pose: ",new_pose)
    # sequence_=["xyz","xzy","yxz","yzx","zxy","zyx","XYZ","XZY","YXZ","YZX","ZXY","ZYX"]
    # for seq0 in sequence_:
    #     for seq1 in sequence_:
    #         # print(seq1)
    #         for seq2 in sequence_:
    #             new_pose=functions_.posetrans3(current_pose,[0,0,0,0,0,0],seq0=seq0,seq1=seq1,seq2=seq2)
    #             magn=np.linalg.norm(np.array(new_pose[3:6])-np.array(current_pose[3:6]))
    #             # print(magn)
    #             if np.linalg.norm(np.array(new_pose[3:6])-np.array(current_pose[3:6]))<10:
    #                 print("true for seq1 = ",seq1," and seq2 = ",seq2)
                # print("current_pose: ",current_pose)
                # new_pose=[round(num, 3) for num in new_pose]
                # print("new_pose:     ",new_pose)
    # current_pose_test=list(map(add,current_pose,[1,1,1,1,1,1]))
    # magn=np.linalg.norm(np.array(current_pose[3:6])-np.array(current_pose_test[3:6]))
    # print(magn)


    highlevel_movements.move_to_waypoint_linear(e, base, base_cyclic, new_pose,speed=0.05)
