#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading
import multiprocessing
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 40

def sinus_move(e,v_max,degrees_move,joint_id,direction,base,base_cyclic):
    start_speed=10
    current_deg_init=get_joint_angles(base_cyclic)
    current_deg_init=current_deg_init[joint_id] #e.g. 330
    print("current_deg_init: ",current_deg_init)

    #make zero
    current_deg = 0
    joint_speed_vals=[0, 0, 0, 0, 0, 0, 0]
    joint_speed_vals[joint_id]=start_speed
    #INSERT send init joint_speed_command
    # if direction==-1:
    #     joint_speed_vals[joint_id]=start_speed
    #     example_send_joint_speeds(e,base,speeds=joint_speed_vals,timer=0)
    # else:
    #     joint_speed_vals[joint_id]=-start_speed
    #     example_send_joint_speeds(e,base,speeds=joint_speed_vals,timer=0)
    # example_send_joint_speeds(e,base,speeds=joint_speed_vals,timer=0)
    while e.is_set()==False:
        time.sleep(0.1)
        current_deg_robot=get_joint_angles(base_cyclic)
        current_deg_robot=current_deg_robot[joint_id] #e.g. 331

        current_deg=abs(current_deg_robot-current_deg_init) #e.g. 1
        trash_delta=abs(current_deg_robot-current_deg_init) #e.g. 1
        if current_deg>180:
            current_deg=abs(current_deg-360)
        if current_deg>degrees_move-1:
            break
        # v_out=0.5*v_max*math.sin(math.radians(current_deg)*2*math.pi/math.radians(degrees_move)-math.pi/2)+0.5*v_max
        # v_out=v_max*math.sin(math.radians(current_deg)*2*math.pi/math.radians(degrees_move*2))
        v_out=v_max*math.sin(math.radians(current_deg)*2*math.pi/math.radians(degrees_move*2))
        #v_out=abs(v_out)
        #print("v_out: ",v_out)
        #v_out=0.5*v_max*math.sin(math.radians(current_deg)*2*math.pi/math.radians(degrees_move*1.1)-0.9*math.pi/2)+0.5*v_max
        # if v_out<=start_speed:
        #     v_out=start_speed
        if direction==-1: #rotate positive
            joint_speed_vals[joint_id]=v_out+start_speed
        else: #rotate negative
            joint_speed_vals[joint_id]=-v_out-start_speed
        example_send_joint_speeds(e,base,speeds=joint_speed_vals,timer=0)
        print("current_deg_robot: ",round(current_deg_robot,3), " current_deg ",round(current_deg,3)," v_out ",round(v_out,3))

    base.Stop()
    #time.sleep(0.5)
    current_deg_robot=get_joint_angles(base_cyclic)
    current_deg_robot=current_deg_robot[joint_id] #e.g. 331
    print("current_deg_robot: ",current_deg_robot)
    #print("wave done")

def get_tcp_pose(base_cyclic):
    feedback=base_cyclic.RefreshFeedback()
    pose=[
        round(feedback.base.tool_pose_x,3),
        round(feedback.base.tool_pose_y,3),
        round(feedback.base.tool_pose_z,3),
        round(feedback.base.tool_pose_theta_x,3),
        round(feedback.base.tool_pose_theta_y,3),
        round(feedback.base.tool_pose_theta_z,3),
        ]
    return pose
def get_joint_angles(base_cyclic):
    feedback = base_cyclic.RefreshFeedback()
    joint_angles=[]
    expected_number_of_actuators=7
    for axis in range(expected_number_of_actuators):
        joint_angles.append(round(feedback.actuators[axis].position,2))
    #from https://github.com/Kinovarobotics/kortex/blob/master/api_python/doc/markdown/messages/BaseCyclic/ActuatorFeedback.md#
    return joint_angles

def convert_speed2time(target_pose,speed,base_cyclic):
    #detect mode
    if len(target_pose)==6: #Cartesian Move
        current_pose=get_tcp_pose(base_cyclic)
        delta_list=[]
        for i0 in range(len(target_pose)):
            a=current_pose[i0]
            b=target_pose[i0]
            delta=abs(b-a)
            delta_list.append(delta)
    else: #angular move
        current_pose=get_joint_angles(base_cyclic)
        #get the delta of the poses
        delta_list=[]
        for i0 in range(len(target_pose)):
            a=current_pose[i0]
            b=target_pose[i0]
            delta=abs(b-a)
            if delta>180:
                delta=abs(delta-360)
            delta_list.append(delta)
    #find the farthest distance
    dist=max(delta_list)
    return dist/speed #time

def send_gripper_command(base,value=0.5):
    # desired opening value 1=close, 0 is open
    opening_val=value
    speed_val_=0.1
    last_gripper_val=666
    t0=time.time()
    #check the finger value
    # Create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_request = Base_pb2.GripperRequest()
    # Wait for reported position to be opened
    gripper_request.mode = Base_pb2.GRIPPER_POSITION
    gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
    if gripper_measure.finger[0].value < opening_val: #close
        speed_val=-speed_val_
        gr_action="close"
    else: #open
        speed_val=speed_val_
        gr_action="open"
    # Create the GripperCommand we will send
    print("moving to ",str(opening_val))
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    # Set speed to open gripper
    if gr_action=="open":
        print ("Opening gripper using speed command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = speed_val
        base.SendGripperCommand(gripper_command)
        gripper_request = Base_pb2.GripperRequest()
        # Wait for reported position to be opened
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        while True:
            gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                act_gripper_val=gripper_measure.finger[0].value
                delta=abs(last_gripper_val-act_gripper_val)
                print("Current pos is : {0}".format(act_gripper_val))
                time.sleep(0.1)
                if delta==0: #gripper cant move anymore
                    break
                elif act_gripper_val < opening_val+0.01 or time.time()-t0>5:
                    break
                last_gripper_val=act_gripper_val
            else: # Else, no finger present in answer, end loop
                break
    else:
        # Set speed to close gripper
        print ("Closing gripper using speed command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = speed_val
        base.SendGripperCommand(gripper_command)
        # Wait for reported speed to be 0
        gripper_request.mode = Base_pb2.GRIPPER_POSITION

        while True:
            gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                act_gripper_val=gripper_measure.finger[0].value
                delta=abs(last_gripper_val-act_gripper_val)
                print("Current pos is : {0}".format(act_gripper_val))
                time.sleep(0.1)
                if delta==0: #gripper cant move anymore
                    break
                elif act_gripper_val > opening_val-0.05:
                    break
                elif time.time()-t0>5:
                    break
                last_gripper_val=act_gripper_val
            else: # Else, no finger present in answer, end loop
                break
    base.Stop()
    return True
def set_joint_speeds(base,j_speed_setting=5):
    #The Function does nothing ...
    #j_speed_setting=5 #degrees/second
    joint_speeds_=Base_pb2.JointSpeeds()
    joint_speeds_.duration=0
    for i0 in range(7):
        axis0=joint_speeds_.joint_speeds.add()
        axis0.joint_identifier=i0
        axis0.value=j_speed_setting
        axis0.duration=0
    base.SendJointSpeedsCommand(joint_speeds_)

def check_for_sequence_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications on a sequence

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification, e = e):
        event_id = notification.event_identifier
        task_id = notification.task_index
        if event_id == Base_pb2.SEQUENCE_TASK_COMPLETED:
            print("Sequence task {} completed".format(task_id))
        elif event_id == Base_pb2.SEQUENCE_ABORTED:
            print("Sequence aborted with error {}:{}"\
                .format(\
                    notification.abort_details,\
                    Base_pb2.SubErrorCodes.Name(notification.abort_details)))
            e.set()
        elif event_id == Base_pb2.SEQUENCE_COMPLETED:
            print("Sequence completed.")
            e.set()
    return check
def call_kinova_web_sequence(base, base_cyclic,seq_name=""):
    #get the Sequence
    sequences = base.ReadAllSequences()
    sequence_handle = None
    for sequence in sequences.sequence_list:
        if sequence.name == seq_name:
            #print(sequence)
            sequence_handle=sequence.handle
            print("Executing Sequence: %s"%(seq_name))
    if sequence_handle == None:
        print("Can't find sequence, check name.")
        sys.exit(0)
    e = threading.Event()
    notification_handle = base.OnNotificationSequenceInfoTopic(
        check_for_sequence_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.PlaySequence(sequence_handle)
    #comes from here: https://github.com/Kinovarobotics/kortex/blob/master/api_cpp/doc/markdown/summary_pages/Base.md
    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if not finished:
        print("Timeout on action notification wait")
    return finished

def euler2mat(rotation=[0,0,0],translation=[0,0,0],point=[0,0,0]): #Zum Verschieben des Tools in gewuenschte Richtung (Point)
    #Muss noch getestet werden
    #Point nicht implementiert
    xC, xS = math.cos(rotation[0]), math.sin(rotation[0])
    yC, yS = math.cos(rotation[1]), math.sin(rotation[1])
    zC, zS = math.cos(rotation[2]), math.sin(rotation[2])
    dX = translation[0]
    dY = translation[1]
    dZ = translation[2]
    Translate_matrix = np.array([[1, 0, 0, dX],[0, 1, 0, dY],[0, 0, 1, dZ],[0, 0, 0, 1]])
    Rotate_X_matrix = np.array([[1, 0, 0, 0],[0, xC, -xS, 0],[0, xS, xC, 0],[0, 0, 0, 1]])
    Rotate_Y_matrix = np.array([[yC, 0, yS, 0],[0, 1, 0, 0],[-yS, 0, yC, 0],[0, 0, 0, 1]])
    Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],[zS, zC, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    #matrix =np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))
    matrix =np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))
    new_point=np.dot(matrix,[point[0],point[1],point[2],1])
    if point[0]+point[1]+point[2]!=0:
    	return new_point
    else:
    	return matrix

def tool_twist_time(e,base,timer=3,pose_distance=[0,-0.2,0,0,0,0]):
    #works like this: within 3 seconds move 20cm in negative y direction
    #when this function is called, the robot moves unlimited into the
    #pose_increment directions, where the valuas are the speed and direction(+/-)
    #the movement can be stopped with base.Stop()

    command = Base_pb2.TwistCommand()

    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    command.duration = 0

    twist = command.twist
    twist.linear_x = pose_distance[0]/timer #meters/second
    twist.linear_y = pose_distance[1]/timer #meters/second
    twist.linear_z = pose_distance[2]/timer #meters/second
    twist.angular_x = pose_distance[3]/timer #deg/second
    twist.angular_y = pose_distance[4]/timer #deg/second
    twist.angular_z = pose_distance[5]/timer #deg/second
    base.SendTwistCommand(command)

    # Let time for twist to be executed
    #time.sleep(5)
    t0=time.time()
    while e.is_set()==False and time.time()-t0<timer:
        time.sleep(0.05)

    base.Stop()


    #print ("Stopping the robot...")
    #base.Stop()
    #time.sleep(1)

    return True
def example_twist_command(base,pose_increment=[0,0,0,0,0,0]):
    #when this function is called, the robot moves unlimited into the
    #pose_increment directions, where the valuas are the speed and direction(+/-)
    #the movement can be stopped with base.Stop()

    command = Base_pb2.TwistCommand()

    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    command.duration = 0

    twist = command.twist
    twist.linear_x = pose_increment[0] #meters/second
    twist.linear_y = pose_increment[1] #meters/second
    twist.linear_z = pose_increment[2] #meters/second
    twist.angular_x = pose_increment[3] #deg/second
    twist.angular_y = pose_increment[4] #deg/second
    twist.angular_z = pose_increment[5] #deg/second
    base.SendTwistCommand(command)

    # Let time for twist to be executed
    #time.sleep(5)

    #print ("Stopping the robot...")
    #base.Stop()
    #time.sleep(1)

    return True
# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def angular_action_movement(e0,base,base_cyclic,joint_positions=[0,0,0,0,0,0,0],speed=5,watch_for_flag=False,success_flag_shared=multiprocessing.Value("i",0)):
    #At first change the joint speed to the desired value
    #convert desired speed to time depending on the current Position
    movement_time=convert_speed2time(target_pose=joint_positions,speed=speed,base_cyclic=base_cyclic)
    action = Base_pb2.Action()
    action.name = "angular action movement"
    action.application_data = ""
    #joint_speed=action.change_joint_speeds
    actuator_count = base.GetActuatorCount()
    #set speed this method is not working(?)
    angular_speed=action.reach_joint_angles.constraint
    angular_speed.type= 1 # type:speed 1=time in sec 2= deg/sec (asynchronous movements only)
    #https://github.com/Kinovarobotics/kortex/blob/master/api_python/doc/markdown/messages/Base/JointTrajectoryConstraint.md#
    angular_speed.value= movement_time+1 #time or deg/second


    # Move arm to desired angles
    i0=0
    for joint_id in range(7):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = joint_positions[i0]
        i0+=1

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)
    if watch_for_flag==True:
        while success_flag_shared.value==0 or e.is_set()==False:
            time.sleep(0.1)
        finished=True
        base.stop()
        base.Unsubscribe(notification_handle)
    else:
        t0=time.time()
        while e.is_set()==False and time.time()-t0<TIMEOUT_DURATION:
            if e0.is_set()==True: #all processes were terminated from GUI
                base.Stop()
                base.Unsubscribe(notification_handle)
                return False
        finished = e.wait(TIMEOUT_DURATION) #waits until True
        base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    time.sleep(0.2) #damping time
    return finished

def waypoint_trajectory_movement(base, base_cyclic,waypoint,speed=0.1,watch_for_flag=False):
    constrained_pose = Base_pb2.ConstrainedPose()
    constrained_pose.constraint.speed.translation = speed # m/s
    constrained_pose.constraint.speed.orientation = 30 # deg/sec

    cartesian_pose = constrained_pose.target_pose
    cartesian_pose.x = waypoint[0]          # (meters)
    cartesian_pose.y = waypoint[1]          # (meters)
    cartesian_pose.z = waypoint[2]          # (meters)
    cartesian_pose.theta_x = waypoint[3] # (degrees)
    cartesian_pose.theta_y = waypoint[4] # (degrees)
    cartesian_pose.theta_z = waypoint[5] # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Reaching cartesian pose...")
    base.PlayCartesianTrajectory(constrained_pose)

    if watch_for_flag==True:
        while success_flag_shared.value==0 or e.is_set()==False:
            time.sleep(0.1)
        finished=True
        base.stop()
        base.Unsubscribe(notification_handle)
    else:
        t0=time.time()
        while e.is_set()==False and time.time()-t0<TIMEOUT_DURATION:
            if e0.is_set()==True: #all processes were terminated from GUI
                base.Stop()
                base.Unsubscribe(notification_handle)
                return False
        finished = e.wait(TIMEOUT_DURATION) #waits until True
        base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def move_to_waypoint_linear(e0,base, base_cyclic,waypoint,speed=0.1,watch_for_flag=False):
    #Service Dok: https://github.com/Kinovarobotics/kortex/blob/master/api_python/doc/markdown/index.md
    print("Moving to Waypoint ...")

    #speed=Base_pb2.CartesianSpeed()
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""
    #speed.translation = 0.01 #meters per second
    #feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_speed= action.reach_pose.constraint
    cartesian_speed.speed.translation = speed #meters/second
    cartesian_speed.speed.orientation = 5 #deg/second
    cartesian_pose.x = waypoint[0]          # (meters)
    cartesian_pose.y = waypoint[1]    # (meters)
    cartesian_pose.z = waypoint[2]    # (meters)
    cartesian_pose.theta_x = waypoint[3] # (degrees)
    cartesian_pose.theta_y = waypoint[4] # (degrees)
    cartesian_pose.theta_z = waypoint[5] # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)
    if watch_for_flag==True:
        while success_flag_shared.value==0 or e.is_set()==False:
            time.sleep(0.1)
        finished=True
        base.stop()
        base.Unsubscribe(notification_handle)
    else:
        t0=time.time()
        while e.is_set()==False and time.time()-t0<TIMEOUT_DURATION:
            if e0.is_set()==True: #all processes were terminated from GUI
                base.Stop()
                base.Unsubscribe(notification_handle)
                return False
        finished = e.wait(TIMEOUT_DURATION) #waits until True
        base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished


def euler2rotvec(pose):
	if len(pose)==6:
		pose[3:6]=R.from_euler('YXZ', pose[3:6], degrees=False).as_rotvec()
		return pose
	elif len(pose)==3:
		pose=R.from_euler('YXZ', pose, degrees=False).as_rotvec()
		return pose

def example_send_joint_speeds(e0,base,speeds=[-18, 0, 0, 0, 0, 0, 0],timer=10):

    joint_speeds = Base_pb2.JointSpeeds()

    # actuator_count = base.GetActuatorCount().count
    # The 7DOF robot will spin in the same direction for 10 seconds
    # if actuator_count == 7:
    # speeds = [-18, 0, 0, 0, 0, 0, 0]
    i = 0
    for speed in speeds:
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i
        joint_speed.value = speed
        joint_speed.duration = 0
        i = i + 1
    #print ("Sending the joint speeds for 10 seconds...")
    base.SendJointSpeedsCommand(joint_speeds)
    t0=time.time()
    if timer==0:
        return True #robot ist still running and stopped in parent function
    else:
        while time.time()-t0<timer and e0.is_set()==False: #success_flag_shared.value==False:
            time.sleep(0.1)
        print ("Stopping the robot")
        base.Stop()
        return True

def main():
    print("Henlo!")

if __name__ == "__main__":
    main()
