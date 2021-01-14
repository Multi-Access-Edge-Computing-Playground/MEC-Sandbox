import sys
import os
import time
import threading
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
print("UR")
startpose = [0.2,0.5,0.1,1.57,0,0]
point2 = [0.2,0.5,0.6,1.57,0,0]

print("KINOVA")
# options1=["xyz","xzy","yxz","yzx","zxy","zyx", "XYZ","XZY","YXZ","YZX","ZXY","ZYX"]
# options2=["xyz","xzy","yxz","yzx","zxy","zyx", "XYZ","XZY","YXZ","YZX","ZXY","ZYX"]
options1=["zyx"]
options2=["zyx"]
startpose = [0.0,0,0.8,-90.823,171.8,113.73]
point2 = [0,-0.2,1,0,0,0]



print(str(startpose)+" to "+str(point2))
for option1 in options1:
    for option2 in options2:
        As2=R.from_euler(option1,startpose[3:6],degrees=True).as_matrix()
        As2_3_3=R.from_euler(option1,startpose[3:6],degrees=True).as_matrix()
        As2=np.vstack([As2,[0,0,0]])
        #add column with translation
        a=np.array(startpose[0:3]+[1])
        b=a.reshape(-1,1)
        As2=np.append(As2,b,axis=1)
        print(As2)

startpose = [0.0,0,0.8,-90.823,171.8,113.73]
point2 = [0,-0.2,1,0,0,0]
#Rot matrix
D1=R.from_euler("zyx",startpose[3:6],degrees=True).as_matrix()
#Transformat Matrix
D1_4x4=np.vstack([D1,[0,0,0]])
a=np.array(startpose[0:3]+[1])
b=a.reshape(-1,1)
D1_4x4=np.append(D1_4x4,b,axis=1)
P3=np.array(point2[0:3]+[1])
P3=P3.reshape(-1,1)
x=np.dot(D1_4x4,P3)
        #
        #point in As2_ =point2

        As2_=R.from_euler(option1,point2[3:6],degrees=True).as_matrix()
        As2_=np.vstack([As2_,[0,0,0]])
        #add column with translation
        a=np.array(point2[0:3]+[1])
        b=a.reshape(-1,1)
        As2_=np.append(As2_,b,axis=1)
        #print(As2_)

        pose_trans=np.dot(As2,As2_)
        # lets make an inverse transformation
        #inverse As2
        inv_As2=np.linalg.inv(As2)
        As1_=np.dot(inv_As2,pose_trans)
        #Results without inverse Matrix
        new_rot=R.from_matrix(pose_trans[0:3,0:3]).as_euler(option2,degrees=True)
        new_trans=pose_trans[0:3,3]
        new_pose=list(new_trans)+list(new_rot)
        # if new_pose[2]<startpose[2] and new_pose[0]>startpose[0]:
        #     print(option1," and ",option2,": ",new_pose)
        #Results with inverse Matrix
        new_rot=R.from_matrix(As1_[0:3,0:3]).as_euler(option2,degrees=True)
        new_trans=As1_[0:3,3]
        new_pose = list(new_trans)+list(new_rot)
        new_pose = [round(num, 2) for num in new_pose]

        #if new_pose[2]<startpose[2] and new_pose[0]>startpose[0]:
        print("inverse: ",option1," and ",option2,": ",new_pose[0:3])
        a=np.array(point2[0:3]+[1])
        a=a.reshape(-1,1)
        print(b)
        b=np.array(startpose[0:3]+[1])
        b=b.reshape(-1,1)
        print(a)
        x = np.dot(inv_As2,a)-b
        print(x)
        inv_As2_3_3=np.linalg.inv(As2_3_3)
        print(np.dot(inv_As2,b))
        x = np.dot(inv_As2_3_3,point2[0:3])+startpose[0:3]
        print(x)
        x = np.dot(inv_As2_3_3,startpose[0:3])- point2[0:3]
        print(x)
