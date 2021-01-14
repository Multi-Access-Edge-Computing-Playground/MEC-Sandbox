##read_BOM and create UR-SCRIPT file with Waypoints
#Whats the input [FW5128]
#Whats the output List with Coordinates like [[X,Y,Z,RX,RY,RZ],[X,Y,Z,RX,RY,RZ],[X,Y,Z,RX,RY,RZ],[X,Y,Z,RX,RY,RZ], ...]
#The Original model is in a horizontal position, to make it stand in a vertical position, a rotation y(or x?) is neccessary
#The Z-Direction of Objects is in case of the objects in the opposite direction. me will flip the z axis with "posetrans"

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

def calc_rotmat_to_align_vec_a_to_vec_b(vec_b, vec_a):
	#Lets use a Z-Y-X rotation sequence
	# def s(q):
	# 	q=math.radians(q)
	# 	return math.sin(q)
	# def c(q):
	# 	q=math.radians(q)
	# 	return math.cos(q)
	#move origin KS into vec_a
	#vec_b is further away
	rot_dir=1
	vec_b=list(map(sub, vec_b,vec_a))
	# vec_b=list(map(sub, vec_a,vec_b))
	vec_a=[0,0,0]

	if vec_b[0]>vec_a[0] and vec_b[1]<vec_a[1]: #Quadrant I
		try:
			alpha=rot_dir*(math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])))))
		except:
			alpha=rot_dir*(90)
	elif vec_b[0]>vec_a[0] and vec_b[1]>vec_a[1]: #Quadrant II
		try:
			alpha=rot_dir*(-math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1]))))+180)
		except:
			alpha=rot_dir*(90+180)
	elif vec_b[0]<vec_a[0] and vec_b[1]>vec_a[1]: #Quadrant III
		try:
			alpha=rot_dir*(math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1]))))-180)
		except:
			alpha=rot_dir*(90-180)
	elif vec_b[0]<vec_a[0] and vec_b[1]<vec_a[1]: #Quadrant I
		try:
			alpha=rot_dir*(-math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])))))
		except:
			alpha=rot_dir*(-90)
	#perform first rotation
	rot_mat=R.from_euler('ZYX', [0, 0, alpha], degrees=True).as_matrix()

	try:
		beta= 0
	except:
		beta=0
	try:
		gamma= rot_dir*(math.degrees(math.atan((vec_b[1]-vec_a[1])/(vec_b[2]-vec_a[2]))))
	except:
		gamma=rot_dir*(90)
	print("alpha: ",alpha)
	print("beta: ",beta)
	print("gamma: ",gamma)
	# rot_z= np.array([
	# 	[ c(alpha),-s(alpha),0],
	# 	[s(alpha),c(alpha),0],
	# 	[0,0,1,]
	# 	])
	# rot_y= np.array([
	# 	[ c(beta),0,s(beta)],
	# 	[0,1,0],
	# 	[-s(beta),0,c(beta),]
	# 	])
	# rot_x= np.array([
	# 	[1,0,0],
	# 	[0, c(gamma),-s(gamma)],
	# 	[0,s(gamma),c(gamma)]
	# 	])
	# # rot_mat=np.matmul(rot_x,np.matmul(rot_y,rot_z))
	# # rot_mat=np.matmul(rot_z,np.matmul(rot_y,rot_x))
	# r = R.from_euler('zyx', [
	# 	[gamma, 0, 0],
	# 	[0, beta, 0],
	# 	[0, 0, alpha]], degrees=True)
	# r.as_quat()beta
	# rot_mat=rot_mat.apply([1,1,1])
	# rot_mat=r.as_matrix()
	# quanternions=r.as_quat()
	# rot_mat=R.from_quat(quanternions).as_matrix()
	rot_mat=R.from_euler('ZYX', [gamma, beta, alpha], degrees=True).as_matrix()
	# rot_mat=R.from_euler('ZYX', [alpha, beta, gamma], degrees=True).as_matrix()
	print(rot_mat)
	return rot_mat
def posetrans(Startpose,Translation=[0,0,0],Rotation=[0,0,0]):
	if len(Translation)==3: #backwards compatibility
		trans_pose=Translation+Rotation
	else:
		trans_pose=Translation #backwards compatibility
	start_pose=Startpose #backwards compatibility
	start_pose_rot_mat=R.from_rotvec(start_pose[3:6]).as_matrix()
	start_pose_rot_mat=np.vstack([start_pose_rot_mat,[0,0,0]])
	start_pose_trans=np.array(start_pose[0:3]+[1])
	start_pose_trans=start_pose_trans.reshape(-1,1)
	start_pose_rot_mat=np.append(start_pose_rot_mat,start_pose_trans,axis=1)

	trans_pose_rot_mat=R.from_rotvec(trans_pose[3:6]).as_matrix()
	trans_pose_rot_mat=np.vstack([trans_pose_rot_mat,[0,0,0]])
	trans_pose_trans=np.array(trans_pose[0:3]+[1])
	trans_pose_trans=trans_pose_trans.reshape(-1,1)
	trans_pose_rot_mat=np.append(trans_pose_rot_mat,trans_pose_trans,axis=1)

	new4x4=np.dot(start_pose_rot_mat,trans_pose_rot_mat)
	new_rot=R.from_matrix(new4x4[0:3,0:3]).as_rotvec()
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
	Axes3D.plot()
def main():

	extracted_waypoints_rel=[[0,0,0,0,0,0],
		[0.8,-0.202,0.404,0,0,0],[0.5,-0.2,0.4,0,0,0],
		[0.8,-0.82,0.404,0,0,0],[0.5,-0.52,0.4,0,0,0],
		[0.8,0.82,0.404,0,0,0],[0.5,0.52,0.4,0,0,0],
		[-0.8,-0.82,0.404,0,0,0],[-0.5,-0.52,0.4,0,0,0],
		[-0.8,0.82,0.404,0,0,0],[-0.5,0.52,0.4,0,0,0]]
	# Testing #1 rotate point and calculate again
	rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[1], extracted_waypoints_rel[2])
	rot_rot_vec=list(R.from_matrix(rot_mat).as_rotvec())
	extracted_waypoints_rel[2][3:6]=rot_rot_vec

	rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[3], extracted_waypoints_rel[4])
	rot_rot_vec=list(R.from_matrix(rot_mat).as_rotvec())
	extracted_waypoints_rel[4][3:6]=rot_rot_vec

	rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[5], extracted_waypoints_rel[6])
	rot_rot_vec=list(R.from_matrix(rot_mat).as_rotvec())
	extracted_waypoints_rel[6][3:6]=rot_rot_vec

	rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[7], extracted_waypoints_rel[8])
	rot_rot_vec=list(R.from_matrix(rot_mat).as_rotvec())
	extracted_waypoints_rel[8][3:6]=rot_rot_vec

	rot_mat=calc_rotmat_to_align_vec_a_to_vec_b(extracted_waypoints_rel[9], extracted_waypoints_rel[10])
	rot_rot_vec=list(R.from_matrix(rot_mat).as_rotvec())
	extracted_waypoints_rel[10][3:6]=rot_rot_vec

	#lets add a second point that is pose transed to see the direction of an axis (z)
	x_axis=[]
	y_axis=[]
	z_axis=[]
	for i0 in range(len(extracted_waypoints_rel)):
		z_axis.append(posetrans(extracted_waypoints_rel[i0][0:6],Translation=[0,0,0.1],Rotation=[0,0,0]))
		x_axis.append(posetrans(extracted_waypoints_rel[i0][0:6],Translation=[0.1,0,0],Rotation=[0,0,0]))
		y_axis.append(posetrans(extracted_waypoints_rel[i0][0:6],Translation=[0,0.1,0],Rotation=[0,0,0]))
		#print(pose)
	print(extracted_waypoints_rel[0])
	draw_axis(extracted_waypoints_rel,x_axis,y_axis,z_axis)



if __name__ == '__main__':
	main()
