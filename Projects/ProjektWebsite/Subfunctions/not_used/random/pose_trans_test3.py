import sys
import os
import time
import threading
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
startpose = [0.0,0,0.8,-90.823,171.8,113.73]
point2 = [0,0,0.5,0,0,0]
def posetrans_angie(startpose,point2,option="yzx"):
    #Rot matrix
    options1=["xyz","xzy","yxz","yzx","zxy","zyx", "XYZ","XZY","YXZ","YZX","ZXY","ZYX"]
    # for option in options1:
    D1=R.from_euler(option,startpose[3:6],degrees=True).as_matrix()
    #Transformat Matrix
    D1_4x4=np.vstack([D1,[0,0,0]])
    a=np.array(startpose[0:3]+[1])
    b=a.reshape(-1,1)
    D1_4x4=np.append(D1_4x4,b,axis=1)

    P3=np.array(point2[0:3]+[1])
    P3=P3.reshape(-1,1)
    P3_4x4=R.from_euler(option,point2[3:6],degrees=True).as_matrix()
    P3_4x4=np.vstack([D1,[0,0,0]])
    P3_4x4=np.append(P3_4x4,P3,axis=1)
    #x=np.dot(D1_4x4,P3)
    x=np.dot(D1_4x4,P3_4x4)
    #for option2 in options1:
    new_rot=R.from_matrix(x[0:3,0:3]).as_euler(option,degrees=True)
    new_trans=x[0:3,3]
    new_pose=list(new_trans)+list(new_rot)
    # if new_pose[0]>0 and new_pose[2]>0.6 and new_pose[4]:
        #print(option,"\n", x)
        #of interest: yzx, YXZ
    print(option,"\n",new_pose)
    return new_pose
#def posetrans_angie2(startpose,point2,option="yzx"):
#Rot matrix
option="yzx"
options1=["xyz","xzy","yxz","yzx","zxy","zyx", "XYZ","XZY","YXZ","YZX","ZXY","ZYX"]
# for option in options1:
D1=R.from_euler(option,startpose[3:6],degrees=True).as_matrix()
#Transformat Matrix
D1_4x4=np.vstack([D1,[0,0,0]])
a=np.array(startpose[0:3]+[1])
b=a.reshape(-1,1)
D1_4x4=np.append(D1_4x4,b,axis=1)

inv_D1_4x4=np.linalg.inv(D1_4x4)

P3=np.array(point2[0:3]+[1])
P3=P3.reshape(-1,1)
P3_4x4=R.from_euler(option,point2[3:6],degrees=True).as_matrix()
P3_4x4=np.vstack([D1,[0,0,0]])
P3_4x4=np.append(P3_4x4,P3,axis=1)
#x=np.dot(D1_4x4,P3)
x=np.dot(D1_4x4,P3_4x4)
inv_x=np.linalg.inv(x)
p3=np.dot(inv_x,[[0],[0],[0],[1]])
#for option2 in options1:
new_rot=R.from_matrix(x[0:3,0:3]).as_euler(option,degrees=True)
new_trans=x[0:3,3]
new_pose=list(new_trans)+list(new_rot)
# if new_pose[0]>0 and new_pose[2]>0.6 and new_pose[4]:
    #print(option,"\n", x)
    #of interest: yzx, YXZ
print(option,"\n",new_pose)
print(p3)
#return new_pose
# print("NEXT")
#
# #D2*P2
# D2=R.from_euler("zyx",point2[3:6],degrees=True).as_matrix()
# D2_4x4=np.vstack([D2,[0,0,0]])
# a=np.array(point2[0:3]+[1])
# b=a.reshape(-1,1)
# D2_4x4=np.append(D2_4x4,b,axis=1)
#
# P2=np.array(startpose[0:3]+[1])
# P2=P2.reshape(-1,1)
# x=np.dot(D2_4x4,P2)
# print(x)
