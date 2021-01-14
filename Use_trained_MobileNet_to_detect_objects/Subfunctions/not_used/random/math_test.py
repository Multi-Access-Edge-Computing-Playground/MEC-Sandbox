import numpy as np
import math
#3 Points form a plain
A=np.array([3,1,1])
B=np.array([1,4,2])
C=np.array([1,3,4])
middle_of_bbox=np.array([1,3.5,3])
n_dir_vector=np.cross(B-A,C-A)

distance=1 #meters
lambda_= distance/np.linalg.norm(n_dir_vector)
point=middle_of_bbox+lambda_*n_dir_vector
print("point pos: ",point)
lambda_= -distance/np.linalg.norm(n_dir_vector)
point=middle_of_bbox+lambda_*n_dir_vector
print("point neg: ",point)
