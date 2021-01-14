#this script simulates the robot behaviour from logging info
# It might be that the kinova works with a left handed coordinate system
# https://stackoverflow.com/questions/31191752/right-handed-euler-angles-xyz-to-left-handed-euler-angles-xyz
##read_BOM and create UR-SCRIPT file with Waypoints
#Whats the input [FW5128]
#Whats the output List with Coordinates like [[X,Y,Z,RX,RY,RZ],[X,Y,Z,RX,RY,RZ],[X,Y,Z,RX,RY,RZ],[X,Y,Z,RX,RY,RZ], ...]
#The Original model is in a horizontal position, to make it stand in a vertical position, a rotation y(or x?) is neccessary
#The Z-Direction of Objects is in case of the objects in the opposite direction. me will flip the z axis with "posetrans"

#Next Step:
"""
use the Rot_z matrix to rotate p2 (p1 is the origin), then calculate beta&gamma for p1
Rot_x*Rot_y*Rot_z is the solution!
-Generate Rot_z with atan..
-apply mat to p2
-Generate Rot_y with atan..
-apply mat to p2
-Generate Rot_x with atan..
-perform matmul(X,(Y,Z)) and convert to euler "ZYX"
"""

import csv
import sys
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from operator import sub
# from functions_ import calc_rotmat_to_align_vec_a_to_vec_b

def calc_rotmat_to_align_vec_a_to_vec_b(vec_b, vec_a,hand="right",rot_dir=1,dash=True,quadrants=True,encoder="zyx"):
	# Lets use a Z-Y-X rotation sequence to find the angles which describe
	# a rotation matrix that makes z-axis of point a point towards point b,
	# while keeping y-axis (camera) horizontal
	def s(q):
		q=math.radians(q)
		return math.sin(q)
	def c(q):
		q=math.radians(q)
		return math.cos(q)
	#move origin coordinate system into vec_a
	#vec_b is further away
	# rot_dir=1
	vec_b=list(map(sub, vec_b,vec_a))
	vec_a=[0,0,0,0,0,0]

	#calculate the first rotation (alpha)
	if quadrants==True:
		if vec_b[0]>0 and vec_b[1]<0: #Quadrant I
			quad="I"
			try:
				# alpha=rot_dir*(math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])))))
				alpha=rot_dir*(-math.degrees(math.atan((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])))+0)
			except:
				alpha=rot_dir*(90)
		elif vec_b[0]>0 and vec_b[1]>0: #Quadrant II
			quad="II"
			try:
				# alpha=rot_dir*(-math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1]))))+180)
				alpha=rot_dir*(math.degrees(math.atan((vec_b[1]-vec_a[1])/(vec_b[0]-vec_a[0])))+90)
			except:
				alpha=rot_dir*(90+90)
		elif vec_b[0]<0 and vec_b[1]>0: #Quadrant III
			quad="III"
			try:
				# alpha=rot_dir*(math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1]))))-180)
				alpha=rot_dir*(-math.degrees(math.atan((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])))+180)
			except:
				alpha=rot_dir*(90+180)
		elif vec_b[0]<0 and vec_b[1]<0: #Quadrant IV
			quad="IV"
			try:
				# alpha=rot_dir*(-math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])))))
				alpha=rot_dir*(math.degrees(math.atan((vec_b[1]-vec_a[1])/(vec_b[0]-vec_a[0])))+270)
			except:
				alpha=rot_dir*(360)
	else:
		try:
			alpha=rot_dir*(math.degrees(math.atan((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1]))))
		except:
			if (vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])>0:
				alpha=rot_dir*(90)
			else:
				alpha=rot_dir*(-90)
	if dash!=True:
		rot_z_dash= np.array([
			[ c(alpha),-s(alpha),0],
			[s(alpha),c(alpha),0],
			[0,0,1,]
			])
	if dash==True:
		rot_z_dash=np.array([
			[ c(-alpha),-s(-alpha),0],
			[s(-alpha),c(-alpha),0],
			[0,0,1,]
			])
	# print("---rot_z-----")
	# print("vec_b[0]-vec_a[0]: ",vec_b[0]-vec_a[0])
	# print("vec_b[1]-vec_a[1]: ",vec_b[1]-vec_a[1])
	# print("alpha: ",alpha)
	# print("quadrant: ",quad)

	#rotate vec_b with rot_z but in different direction (rot_z') in order to
	#calculate the next rotation angle
	vec_b=np.matmul(rot_z_dash,vec_b[0:3]) #new pose
	print("vec_b': ",vec_b)
	# return rot_z

	#calculate new angle (well, no need to rotate y-axis, because it shall be horizontal)
	try:
		beta= 0
	except:
		beta=0
	# rot_y= np.array([
	# 	[ c(beta),0,s(beta)],
	# 	[0,1,0],
	# 	[-s(beta),0,c(beta),]
	# 	])
	# rot_y_dash= np.array([
	# 	[ c(-beta),0,s(-beta)],
	# 	[0,1,0],
	# 	[-s(-beta),0,c(-beta),]
	# 	])
	#rotate vec_b with rot_y_dash
	# vec_b=np.matmul(rot_y_dash,vec_b[0:3]) #new pose

	#calculate last angle
	if quadrants==True:
		if vec_b[2]>0 and vec_b[1]<0: #Quadrant I
			quad="I"
			try:
				gamma= rot_dir*(-math.degrees(math.atan((vec_b[1]-vec_a[1])/(vec_b[2]-vec_a[2])))+0)
			except:
				gamma=rot_dir*(90)
				# print("exception, using default 90deg")
		elif vec_b[2]<0 and vec_b[1]<0: #Quadrant IV
			quad="IV"
			try:
				gamma= rot_dir*(math.degrees(math.atan((vec_b[2]-vec_a[2])/(vec_b[1]-vec_a[1])))+90)
			except:
				gamma=rot_dir*(90+90)
				# print("exception, using default 90+90deg")
		else:
			gamma = rot_dir*(math.degrees(math.atan((vec_b[1]-vec_a[1])/(vec_b[2]-vec_a[2]))))
	else:
		try:
			gamma = rot_dir*(math.degrees(math.atan((vec_b[1]-vec_a[1])/(vec_b[2]-vec_a[2]))))
		except:
			if (vec_b[1]-vec_a[1])/(vec_b[2]-vec_a[2])>0:
				gamma=rot_dir*(90)
			else:
				gamma=rot_dir*(-90)

	#------------------------------

	# print("---rot_x-----")
	# print("vec_b[1]-vec_a[1]: ",vec_b[1]-vec_a[1])
	# print("vec_b[2]-vec_a[2]: ",vec_b[2]-vec_a[2])
	# print("gamma: ",gamma)
	# print("")
	# print("quadrant: ",quad,"\n")
	# rot_x= np.array([
	# 	[1,0,0],
	# 	[0, c(gamma),-s(gamma)],
	# 	[0,s(gamma),c(gamma)]
	# 	])
	# print("alpha: ",alpha)
	# print("beta: ",beta)
	# print("gamma: ",gamma)

	#calculate rotation matrix
	if hand=="right":
	# for right handed system
		rot_mat=R.from_euler(encoder,[alpha,beta,gamma],degrees=True).as_matrix()
		print("[alpha,beta,gamma]: ",[alpha,beta,gamma])
		angles=[alpha,beta,gamma]
	elif hand=="left":
	# for left handed system
		rot_mat=R.from_euler(encoder,[-alpha,-beta,gamma],degrees=True).as_matrix()
		print("[alpha,beta,gamma]: ",[-alpha,-beta,gamma])
		angles=[-alpha,-beta,gamma]
	return rot_mat

def posetrans(Startpose,Translation=[0,0,0],Rotation=[0,0,0]):
	if len(Translation)==3: #backwards compatibility
		trans_pose=Translation+Rotation
	else:
		trans_pose=Translation #backwards compatibility
	start_pose=Startpose #backwards compatibility
	# start_pose_rot_mat=R.from_rotvec(start_pose[3:6]).as_matrix()
	start_pose_rot_mat=R.from_euler("ZYX",start_pose[3:6],degrees=True).as_matrix()
	start_pose_rot_mat=np.vstack([start_pose_rot_mat,[0,0,0]])
	start_pose_trans=np.array(start_pose[0:3]+[1])
	start_pose_trans=start_pose_trans.reshape(-1,1)
	start_pose_rot_mat=np.append(start_pose_rot_mat,start_pose_trans,axis=1)

	# trans_pose_rot_mat=R.from_rotvec(trans_pose[3:6]).as_matrix()
	trans_pose_rot_mat=R.from_euler("ZYX",trans_pose[3:6],degrees=True).as_matrix()
	trans_pose_rot_mat=np.vstack([trans_pose_rot_mat,[0,0,0]])
	trans_pose_trans=np.array(trans_pose[0:3]+[1])
	trans_pose_trans=trans_pose_trans.reshape(-1,1)
	trans_pose_rot_mat=np.append(trans_pose_rot_mat,trans_pose_trans,axis=1)

	new4x4=np.dot(start_pose_rot_mat,trans_pose_rot_mat)
	new_rot=R.from_matrix(new4x4[0:3,0:3]).as_euler("ZYX",degrees=True)
	new_trans=new4x4[0:3,3]
	new_pose=list(new_trans)+list(new_rot)
	return new_pose


def draw_axis(extracted_waypoints_rel,x_axis,y_axis,z_axis):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	size=0.8
	ax.set_xlim3d(-size, size)
	ax.set_ylim3d(-size, size)
	ax.set_zlim3d(-size, size)
	for i in range(len(extracted_waypoints_rel)):
		VecStart_x=extracted_waypoints_rel[i][0]
		VecStart_y=extracted_waypoints_rel[i][1]
		VecStart_z=extracted_waypoints_rel[i][2]
		VecEnd_x=x_axis[i][0]
		VecEnd_y=x_axis[i][1]
		VecEnd_z=x_axis[i][2]
		ax.plot([VecStart_x, VecEnd_x], [VecStart_y,VecEnd_y],zs=[VecStart_z,VecEnd_z],c="r")
	for i in range(len(extracted_waypoints_rel)):
		VecStart_x=extracted_waypoints_rel[i][0]
		VecStart_y=extracted_waypoints_rel[i][1]
		VecStart_z=extracted_waypoints_rel[i][2]
		VecEnd_x=y_axis[i][0]
		VecEnd_y=y_axis[i][1]
		VecEnd_z=y_axis[i][2]
		ax.plot([VecStart_x, VecEnd_x], [VecStart_y,VecEnd_y],zs=[VecStart_z,VecEnd_z],c="g")
	for i in range(len(extracted_waypoints_rel)):
		VecStart_x=extracted_waypoints_rel[i][0]
		VecStart_y=extracted_waypoints_rel[i][1]
		VecStart_z=extracted_waypoints_rel[i][2]
		VecEnd_x=z_axis[i][0]
		VecEnd_y=z_axis[i][1]
		VecEnd_z=z_axis[i][2]
		ax.plot([VecStart_x, VecEnd_x], [VecStart_y,VecEnd_y],zs=[VecStart_z,VecEnd_z],c="b")
	plt.show()
	# Axes3D.plot()
def main():

	extracted_waypoints_rel_all=[] #test waypoints
	#1st test
	#waypoint 1 + 2 are from the robot (endpose, orthogonal pose)
	extracted_waypoints_rel=[[0,0,0,0,0,0],[0.753,-0.397,0.425,0,0,0],[0.632,-0.392,0.452,0,0,0]]
	#soll_value is the angle taken from the kinova web-app of the orthogonal pose
	soll_value=[90.128,0.478,62.117]

	#2nd test SUCCESS: magn:0.23
	#waypoint 1 + 2 are from the robot (endpose, orthogonal pose)
	# [0.743, -0.42, 0.433, 89.095, 0.115, 62.531]
	# [0.516, -0.303, 0.429, 89.087, 0.118, 62.529]
	extracted_waypoints_rel=[[0,0,0,0,0,0],[0.743, -0.42, 0.433,0,0,0],[0.516, -0.303, 0.429,0,0,0]]
	#soll_value is the angle taken from the kinova web-app of the orthogonal pose
	soll_value=[89.095, 0.115, 62.531]

	#3rd test magn=9
	[0.24, -0.527, 0.969, 45.902, 6.793, 16.448] #final
	[0.177, -0.379, 0.815, 45.894, 6.796, 16.448] #orthogonal
	extracted_waypoints_rel=[[0,0,0,0,0,0],[0.24, -0.527, 0.969,0,0,0],[0.177, -0.379, 0.815,0,0,0]]
	#soll_value is the angle taken from the kinova web-app of the orthogonal pose
	soll_value=[45.902, 6.793, 16.448]

	#4th test magn:0.82
	[0.782, 0.231, 0.365, 88.919, 0.817, 91.748]
	[0.519, 0.223, 0.36, 88.912, 0.821, 91.748]
	extracted_waypoints_rel=[[0,0,0,0,0,0],[0.782, 0.231, 0.365,0,0,0],[0.519, 0.223, 0.36,0,0,0]]
	#soll_value is the angle taken from the kinova web-app of the orthogonal pose
	soll_value=[88.919, 0.817, 91.748]

	""" RESULTS
	hand=right dash=True param=XYZ rot_dir=1 quadr=True magn:0.82 encoder:zyx delta soll-ist:  [0.007638765215205012, 0.817, 0.005699833857420344] #####
	right True XYZ 1 quadr=True magn:0.82 encoder:ZYX delta soll-ist:  [0.007638765215205012, 0.817, 0.005699833857420344] #####
	right True XYZ 1 quadr=True magn:0.82 encoder:xyz delta soll-ist:  [0.007638765215205012, 0.817, 0.005699833857420344] #####
	right True XYZ 1 quadr=True magn:0.82 encoder:XYZ delta soll-ist:  [0.007638765215205012, 0.817, 0.005699833857420344] #####"""

	# from_robot_calculated_pose=[0.3562,-0.182+0.2,0.4447,61.33,0,87.1105]
	# extracted_waypoints_rel.append(from_robot_calculated_pose) #add this pose
	# Testing #1 rotate point and calculate again
	hands=["right","left"]
	dashes=[True,False]
	params=["zyx","ZYX","xyz","XYZ"]
	rot_dirs=[1,-1]
	quadrants=[True,False]
	magn=100
	winner=None
	data=[]
	counter=0
	interesting_sets=[]
	encoders=["zyx","ZYX","xyz","XYZ"]
	success_counter=0
	for hand in hands:
		for dash in dashes:
			for param in params:
				for rot_dir in rot_dirs:
					for quadrant in quadrants:
						for encoder in encoders:
							magn_temp=None
							# pose=0
							print("\n#####Testing "+str(hand)+" "+str(dash)+" "+str(param)+" "+str(rot_dir)+" quadr="+str(quadrant)+"encoder:"+str(encoder)+" #####")
							try:
								rot_mat = calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[1], extracted_waypoints_rel[2],hand=hand,dash=dash,rot_dir=rot_dir,quadrants=quadrant)
								rot_rot_vec=list(R.from_matrix(rot_mat).as_euler(param,degrees=True))
								print("rot_rot_vec: ",rot_rot_vec)
								print("rot_rot_vec soll:",soll_value)
								magn_temp=np.linalg.norm(np.array(list(map(sub,soll_value,rot_rot_vec))))
								magn_temp=round(magn_temp,2)
								print("delta soll-ist:  ",list(map(sub,soll_value,rot_rot_vec)))
								print("magn: ",magn_temp)
								if magn_temp<=magn:
									success_counter+=1
									magn=magn_temp
									# extracted_waypoints_rel[2][3:6]=rot_rot_vec
									extracted_waypoints_rel_all.append([extracted_waypoints_rel[0],extracted_waypoints_rel[1],extracted_waypoints_rel[2][0:3]+rot_rot_vec])
									# pose=extracted_waypoints_rel[2]
									interesting_sets.append(counter)
									winner="\n#####WINNER "+str(hand)+" "+str(dash)+" "+str(param)+" "+str(magn)+" quadr="+str(quadrant)+"encoder:"+str(encoder)+" #####"
							except:
								print("Something is wrong with gamma")
							counter+=1
							data.append(str(hand)+" "+str(dash)+" "+str(param)+" "+str(rot_dir)+" quadr="+str(quadrant)+" magn:"+str(magn_temp)+" encoder:"+str(encoder)+" delta soll-ist:  "+str(list(map(sub,soll_value,rot_rot_vec)))+" #####")
	print(winner)
	print("interesting sets:")
	for set in interesting_sets:
		print(data[set])
	print(counter)








	#
	# rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[3], extracted_waypoints_rel[4])
	# rot_rot_vec=list(R.from_matrix(rot_mat).as_euler("zyx",degrees=True))
	# extracted_waypoints_rel[4][3:6]=rot_rot_vec
	#
	# rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[5], extracted_waypoints_rel[6])
	# rot_rot_vec=list(R.from_matrix(rot_mat).as_euler("zyx",degrees=True))
	# extracted_waypoints_rel[6][3:6]=rot_rot_vec
	#
	# rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[7], extracted_waypoints_rel[8])
	# rot_rot_vec=list(R.from_matrix(rot_mat).as_euler("zyx",degrees=True))
	# extracted_waypoints_rel[8][3:6]=rot_rot_vec
	#
	# rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[9], extracted_waypoints_rel[10])
	# rot_rot_vec=list(R.from_matrix(rot_mat).as_euler("zyx",degrees=True))
	# extracted_waypoints_rel[10][3:6]=rot_rot_vec

	# rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[11], extracted_waypoints_rel[12])
	# rot_rot_vec=list(R.from_matrix(rot_mat).as_euler("zyx",degrees=True))
	# extracted_waypoints_rel[12][3:6]=rot_rot_vec

	#lets add a second point that is pose transed to see the direction of an axis (z)

		#print(pose)
	# print(extracted_waypoints_rel[0])
	# print(extracted_waypoints_rel)
	# print(extracted_waypoints_rel_all)
	# extracted_waypoints_rel_all=extracted_waypoints_rel_all[-3:]
	# print(extracted_waypoints_rel_all)
	for extracted_waypoints_rel in extracted_waypoints_rel_all:
		x_axis=[]
		y_axis=[]
		z_axis=[]
		for i0 in range(len(extracted_waypoints_rel)):
			z_axis.append(posetrans(extracted_waypoints_rel[i0][0:6],Translation=[0,0,0.1],Rotation=[0,0,0]))
			x_axis.append(posetrans(extracted_waypoints_rel[i0][0:6],Translation=[0.1,0,0],Rotation=[0,0,0]))
			y_axis.append(posetrans(extracted_waypoints_rel[i0][0:6],Translation=[0,0.1,0],Rotation=[0,0,0]))
		draw_axis(extracted_waypoints_rel,x_axis,y_axis,z_axis)



if __name__ == '__main__':
	main()
