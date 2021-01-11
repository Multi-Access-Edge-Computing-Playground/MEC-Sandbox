# import the necessary packages
import cv2
import imutils
import time
from Subfunctions import highlevel_movements
from operator import add,sub
global log_string
from scipy.spatial.transform import Rotation as R
import numpy as np
# import mgen
import math
global log_string

class Button():
    def __init__(self):
        self.fontStyle = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.7
        self.fontThickness = 2
        self.color = (147,81,20)
        self.button_px_delta_y = 35 #(depends on font)
        self.x_padding = 10
        self.y_padding = 5
        self.x_bbox_padding = 10
        self.y_bbox_padding = 10

    def insert_button(self,frame,temporary_dict,text,position_id=0,corner="top_left",highlighted=False):
        color=self.color
        fontStyle=self.fontStyle
        fontScale=self.fontScale
        fontThickness=self.fontThickness

        (text_width, text_height), baseline = cv2.getTextSize(text, fontStyle, fontScale, fontThickness)
        if corner=="top_left":
            corner_offset=[0,0]
        elif corner=="top_right":
            height, width = frame.shape[:2]
            corner_offset=[width-(self.x_padding+self.x_bbox_padding+text_width+self.x_bbox_padding+self.x_padding) ,0]
        elif corner=="bottom_left":
            height, width = frame.shape[:2]
            corner_offset=[0,height-(position_id*self.y_padding+position_id*self.button_px_delta_y +2*self.button_px_delta_y+position_id*self.button_px_delta_y+position_id*self.y_padding+self.y_padding)]
        elif corner=="bottom_right":
            height, width = frame.shape[:2]
            corner_offset=[ width-(self.x_padding+self.x_bbox_padding+text_width+self.x_bbox_padding+self.x_padding) ,height-(position_id*self.y_padding+position_id*self.button_px_delta_y +2*self.button_px_delta_y+position_id*self.button_px_delta_y+position_id*self.y_padding+self.y_padding)]
        elif corner=="Arrow":
            height, width = frame.shape[:2]
            # corner_offset=[ width-(self.x_padding+self.x_bbox_padding+text_width+self.x_bbox_padding+self.x_padding) ,height-(position_id*self.y_padding+position_id*self.button_px_delta_y +2*self.button_px_delta_y+position_id*self.button_px_delta_y+position_id*self.y_padding+self.y_padding)]
            if position_id==0:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+0*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-1*self.button_px_delta_y]
            if position_id==1:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+1*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-1*self.button_px_delta_y]
            if position_id==2:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+2*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-1*self.button_px_delta_y]
            if position_id==3:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+0*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-2*self.button_px_delta_y]
            if position_id==4:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+1*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-2*self.button_px_delta_y]
            if position_id==5:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+2*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-2*self.button_px_delta_y]
            if position_id==6:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+0*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-3*self.button_px_delta_y]
            if position_id==7:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+1*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-3*self.button_px_delta_y]
            if position_id==8:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+2*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-3*self.button_px_delta_y]
            if position_id==9:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+0*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-4*self.button_px_delta_y]
            if position_id==10:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+1*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-4*self.button_px_delta_y]
            if position_id==11:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+2*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-4*self.button_px_delta_y]
            if position_id==12:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+0*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-5*self.button_px_delta_y]
            if position_id==13:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+1*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-5*self.button_px_delta_y]
            if position_id==14:
                corner_offset=[int(width/2)-int(1.5*(self.x_bbox_padding+text_width+self.x_bbox_padding))+2*(self.x_bbox_padding+text_width+self.x_bbox_padding),
                    height-2*self.y_padding-5*self.button_px_delta_y]

        #bounding box dimensions
        # print(corner_offset)
        if corner=="Arrow":
            x1_bb = corner_offset[0]+ self.x_padding
            y1_bb = corner_offset[1]+ self.y_padding
            x2_bb = corner_offset[0]+ self.x_padding+self.x_bbox_padding+text_width+self.x_bbox_padding
            y2_bb = y1_bb + self.button_px_delta_y
        else:
            x1_bb = corner_offset[0]+ self.x_padding
            y1_bb = corner_offset[1]+ position_id*self.y_padding+position_id*self.button_px_delta_y +self.button_px_delta_y
            x2_bb = corner_offset[0]+ self.x_padding+self.x_bbox_padding+text_width+self.x_bbox_padding
            # y2_bb = corner_offset[1]+ position_id*self.y_padding+position_id*self.button_px_delta_y +2*self.button_px_delta_y
            y2_bb = y1_bb + self.button_px_delta_y
        #text start coordinate
        x1_tx, y1_tx =x1_bb+self.x_bbox_padding,y2_bb-self.y_bbox_padding

        if highlighted==True:
            cv2.rectangle(frame,(x1_bb,y1_bb),(x2_bb,y2_bb),(255,255,255),-1)
        else:
            cv2.rectangle(frame,(x1_bb,y1_bb),(x2_bb,y2_bb),color,1)
        cv2.putText(frame, text, (x1_tx,y1_tx),fontStyle, fontScale, color, fontThickness)
        temporary_dict[text] = {
            "rectangle" : [x1_bb,y1_bb,x2_bb,y2_bb],
            "on_click" : text,
            "priority" : 2, # if an object is behind a button,
                            # the button is prioritized in GUI process
            }
        return frame, temporary_dict
    def insert_arrows(self,rgb_frame, temporary_dict,highlighted=None):
        commands=["    left   ","   down    ","   right   ",
                  "rot -90 deg","    up     ","rot +90 deg",
                  "  forward  ","   HOME    "," backward  ",
                  " TRANSPORT ","HOME-revers","  Gripper  ",
                  " Look Down "," Chg. Graph","  Look Up  "]
        for i0 in range(len(commands)):
            if i0==highlighted:
                rgb_frame, temporary_dict = self.insert_button(
                            rgb_frame, temporary_dict, commands[i0],i0,"Arrow",highlighted=True)
            else:
                rgb_frame, temporary_dict = self.insert_button(
                            rgb_frame, temporary_dict, commands[i0],i0,"Arrow",highlighted=False)
        return rgb_frame, temporary_dict

def calc_rotmat_to_align_vec_a_to_vec_b(vec_a, vec_b):
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
    rot_dir=1
    vec_b=list(map(sub, vec_b,vec_a))
    vec_a=[0,0,0,0,0,0]

    #calculate the first rotation (alpha)
    if vec_b[0]>vec_a[0] and vec_b[1]<vec_a[1]: #Quadrant I
        quad="I"
        try:
            alpha=rot_dir*(math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])))))
        except:
            alpha=rot_dir*(90)
    elif vec_b[0]>vec_a[0] and vec_b[1]>vec_a[1]: #Quadrant II
        quad="II"
        try:
            alpha=rot_dir*(-math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1]))))+180)
        except:
            alpha=rot_dir*(90+180)
    elif vec_b[0]<vec_a[0] and vec_b[1]>vec_a[1]: #Quadrant III
        quad="III"
        try:
            alpha=rot_dir*(math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1]))))-180)
        except:
            alpha=rot_dir*(90-180)
    elif vec_b[0]<vec_a[0] and vec_b[1]<vec_a[1]: #Quadrant IV
        quad="IV"
        try:
            alpha=rot_dir*(-math.degrees(math.atan(abs((vec_b[0]-vec_a[0])/(vec_b[1]-vec_a[1])))))
        except:
            alpha=rot_dir*(-90)

    # rot_z= np.array([
    #     [ c(alpha),-s(alpha),0],
    #     [s(alpha),c(alpha),0],
    #     [0,0,1,]
    #     ])
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
    #     [ c(beta),0,s(beta)],
    #     [0,1,0],
    #     [-s(beta),0,c(beta),]
    #     ])
    # rot_y_dash= np.array([
    #     [ c(-beta),0,s(-beta)],
    #     [0,1,0],
    #     [-s(-beta),0,c(-beta),]
    #     ])
    #rotate vec_b with rot_y_dash
    # vec_b=np.matmul(rot_y_dash,vec_b[0:3]) #new pose

    #calculate last angle
    if vec_b[2]>vec_a[2] and vec_b[1]<vec_a[1]: #Quadrant I
        quad="I"
        try:
            gamma= rot_dir*(math.degrees(math.atan(abs((vec_b[1]-vec_a[1])/(vec_b[2]-vec_a[2])))))
        except:
            gamma=rot_dir*(90)
            # print("exception, using default 90deg")
    elif vec_b[2]<vec_a[0] and vec_b[1]<vec_a[1]: #Quadrant IV
        quad="IV"
        try:
            gamma= rot_dir*(90+math.degrees(math.atan(abs((vec_b[2]-vec_a[2])/(vec_b[1]-vec_a[1])))))
        except:
            gamma=rot_dir*(90+90)
            # print("exception, using default 90+90deg")

    #------------------------------

    # print("---rot_x-----")
    # print("vec_b[1]-vec_a[1]: ",vec_b[1]-vec_a[1])
    # print("vec_b[2]-vec_a[2]: ",vec_b[2]-vec_a[2])
    # print("gamma: ",gamma)
    # print("")
    # print("quadrant: ",quad,"\n")
    # rot_x= np.array([
    #     [1,0,0],
    #     [0, c(gamma),-s(gamma)],
    #     [0,s(gamma),c(gamma)]
    #     ])
    # print("alpha: ",alpha)
    # print("beta: ",beta)
    # print("gamma: ",gamma)

    #calculate rotation matrix
    rot_mat=R.from_euler("zyx",[alpha,beta,gamma],degrees=True).as_matrix()
    return rot_mat #handing over the angles might be faster...

def forward_kin(joint_angles):
    q1,q2,q3,q4,q5,q6,q7 = joint_angles
    def s(q):
        q=math.radians(q)
        return math.sin(q)
    def c(q):
        q=math.radians(q)
        return math.cos(q)
    T1= np.array([
        [ c(q1),-s(q1),0,0],
        [-s(q1),-c(q1),0,0],
        [0,0,-1,0.1564],
        [0,0,0,1]
        ])
    T2= np.array([
        [ c(q2),-s(q2),0,0],
        [0,0,-1,0.0054],
        [ s(q2),c(q2),0,-0.1284],
        [0,0,0,1]
        ])
    T3= np.array([
        [ c(q3),-s(q3),0,0],
        [0, 0, 1, -0.2104],
        [ -s(q3),-c(q3),0,-0.0064],
        [0,0,0,1]
        ])
    T4= np.array([
        [ c(q4),-s(q4),0,0],
        [0, 0, -1, -0.0064],
        [ s(q4),c(q4),0,-0.2104],
        [0,0,0,1]
        ])
    T5= np.array([
        [ c(q5),-s(q5),0,0],
        [0, 0, 1, -0.2084],
        [ -s(q5),-c(q5),0,-0.0064],
        [0,0,0,1]
        ])
    T6= np.array([
        [ c(q6),-s(q6),0,0],
        [0, 0, -1, 0],
        [ s(q6),c(q6),0,-0.1059],
        [0,0,0,1]
        ])
    T7= np.array([
        [ c(q7),-s(q7),0,0],
        [0, 0, 1, -0.1059],
        [ -s(q7),-c(q7),0,0],
        [0,0,0,1]
        ])
    Ttool= np.array([
        [ c(0),-s(0),0,0], #[1, 0, 0, 0],
        [-s(0),-c(0),0,+0.013], #[0, -1, 0, 0],
        [0,0,-1,-0.17-0.012],
        [0,0,0,1]
        ])
    # PoseTrans=np.array([
    #     [ 1,0,0,0], #[1, 0, 0, 0],
    #     [ 0,1,0,0], #[0, -1, 0, 0],
    #     [ 0,0,1,0.05],
    #     [ 0,0,0,1]
    #     ])

    pose_mat=np.matmul(T1,np.matmul(T2,np.matmul(T3,np.matmul(T4,np.matmul(T5,np.matmul(T6,np.matmul(T7,Ttool)))))))
    # pose_mat=np.matmul(Ttool,np.matmul(T7,np.matmul(T6,np.matmul(T5,np.matmul(T4,np.matmul(T3,np.matmul(T2,T1)))))))
    # print(pose_mat)
    # print(pose_mat[:3,:3])
    pose_translation=list(pose_mat[:-1,3])
    rot_tait_bryan=list(R.from_matrix(pose_mat[:3,:3]).as_euler("xyz",degrees=True))
    # delta_angle=
    return pose_mat#,pose_translation+rot_tait_bryan #(Pose with euler angles)

def forward_kin_plus(joint_angles,camera_shift):
    q1,q2,q3,q4,q5,q6,q7 = joint_angles
    # print("deg: ",camera_shift)
    # camera_shift[3:6]=[math.radians(num) for num in camera_shift[3:6]]
    # print("rad: ",camera_shift)
    def s(q):
        q=math.radians(q)
        return math.sin(q)
    def c(q):
        q=math.radians(q)
        return math.cos(q)
    T1= np.array([
        [ c(q1),-s(q1),0,0],
        [-s(q1),-c(q1),0,0],
        [0,0,-1,0.1564],
        [0,0,0,1]
        ])
    T2= np.array([
        [ c(q2),-s(q2),0,0],
        [0,0,-1,0.0054],
        [ s(q2),c(q2),0,-0.1284],
        [0,0,0,1]
        ])
    T3= np.array([
        [ c(q3),-s(q3),0,0],
        [0, 0, 1, -0.2104],
        [ -s(q3),-c(q3),0,-0.0064],
        [0,0,0,1]
        ])
    T4= np.array([
        [ c(q4),-s(q4),0,0],
        [0, 0, -1, -0.0064],
        [ s(q4),c(q4),0,-0.2104],
        [0,0,0,1]
        ])
    T5= np.array([
        [ c(q5),-s(q5),0,0],
        [0, 0, 1, -0.2084],
        [ -s(q5),-c(q5),0,-0.0064],
        [0,0,0,1]
        ])
    T6= np.array([
        [ c(q6),-s(q6),0,0],
        [0, 0, -1, 0],
        [ s(q6),c(q6),0,-0.1059],
        [0,0,0,1]
        ])
    T7= np.array([
        [ c(q7),-s(q7),0,0],
        [0, 0, 1, -0.1059],
        [ -s(q7),-c(q7),0,0],
        [0,0,0,1]
        ])
    Ttool= np.array([
        [ c(0),-s(0),0,0], #[1, 0, 0, 0],
        [-s(0),-c(0),0,+0.013-0.025-0.04], #[0, -1, 0, 0],
        [0,0,-1,-0.17-0.012+0.165+0],
        [0,0,0,1]
        ])
    #rotate x-axis
    rot_x= np.array([
        [1, 0, 0, 0],
        [0, c(camera_shift[3]), -s(camera_shift[3]), 0],
        [0, s(camera_shift[3]), c(camera_shift[3]), 0],
        [0, 0, 0, 1]
        ])
    # rotate y
    rot_y= np.array([
        [c(camera_shift[4]), 0, s(camera_shift[4]), 0],
        [0, 1, 0, 0],
        [-s(camera_shift[4]), 0, c(camera_shift[4]), 0],
        [0, 0, 0, 1]
        ])
    # rotate z
    rot_z= np.array([
        [c(camera_shift[5]), -s(camera_shift[5]), 0, 0],
        [s(camera_shift[5]), c(camera_shift[5]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])
    #translate into depth
    trans= np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, camera_shift[2]],
        [0, 0, 0, 1]
        ])

    # trans_pose_rot_mat=R.from_euler("zyx",camera_shift[3:6],degrees=False).as_matrix()
    # trans_pose_rot_mat=np.vstack([trans_pose_rot_mat,[0,0,0]])
    # trans_pose_trans=np.array(camera_shift[0:3]+[1])
    # trans_pose_trans=trans_pose_trans.reshape(-1,1)
    # trans_pose_rot_mat=np.append(trans_pose_rot_mat,trans_pose_trans,axis=1)

    trans_pose_rot_mat=np.matmul(
        rot_z,np.matmul(
            rot_y,np.matmul(rot_x,trans)))

    pose_mat=np.matmul(
        T1,np.matmul(
            T2,np.matmul(
                T3,np.matmul(
                    T4,np.matmul(
                        T5,np.matmul(
                            T6,np.matmul(
                                T7,np.matmul(
                                    Ttool,trans_pose_rot_mat))))))))
    # pose_mat=np.matmul(Ttool,np.matmul(T7,np.matmul(T6,np.matmul(T5,np.matmul(T4,np.matmul(T3,np.matmul(T2,T1)))))))
    # print(pose_mat)
    # print(pose_mat[:3,:3])
    pose_translation=list(pose_mat[:-1,3])
    rot_tait_bryan=list(R.from_matrix(pose_mat[:3,:3]).as_euler("zyx",degrees=True))
    # delta_angle=
    return pose_mat,pose_translation+rot_tait_bryan #(Pose with euler angles)

def posetrans(Startpose,Translation=[0,0,0],Rot=[0,0,0]):
    #works with scipy > 1.4.1
    if len(Translation)==3: #backwards compatibility
        trans_pose=Translation+Rot
    else:
        trans_pose=Translation #backwards compatibility
    start_pose=Startpose #backwards compatibility
    start_pose_rot_mat=R.from_euler("ZYX",start_pose[3:6],degrees=True).as_matrix()
    start_pose_rot_mat=np.vstack([start_pose_rot_mat,[0,0,0]])
    start_pose_trans=np.array(start_pose[0:3]+[1])
    start_pose_trans=start_pose_trans.reshape(-1,1)
    start_pose_rot_mat=np.append(start_pose_rot_mat,start_pose_trans,axis=1)
    print(start_pose_rot_mat)

    trans_pose_rot_mat=R.from_euler("ZYX",trans_pose[3:6],degrees=True).as_matrix()
    trans_pose_rot_mat=np.vstack([trans_pose_rot_mat,[0,0,0]])
    trans_pose_trans=np.array(trans_pose[0:3]+[1])
    trans_pose_trans=trans_pose_trans.reshape(-1,1)
    trans_pose_rot_mat=np.append(trans_pose_rot_mat,trans_pose_trans,axis=1)
    print(trans_pose_rot_mat)

    new4x4=np.matmul(start_pose_rot_mat,trans_pose_rot_mat)
    new_rot=R.from_matrix(new4x4[0:3,0:3]).as_euler("ZYX",degrees=True)
    new_trans=new4x4[0:3,3]
    new_pose=list(new_trans)+list(new_rot)
    return new_pose

def pose_comparison(target_pose,base_cyclic,mode):
    if mode=="joint":
        pose=highlevel_movements.get_joint_angles(base_cyclic)
        tolerance=1 #degrees
    elif mode=="cartesian":
        pose=highlevel_movements.get_tcp_pose(base_cyclic)
        tolerance=0.005 #meters
    else:
        raise NameError("mode must be 'joint' or 'cartesian'")
    delta_list=list(map(sub,target_pose,pose))
    for elem in delta_list:
        if abs(elem)>tolerance:
            print("Robot not in position")
            return False
    return True



def draw_loading_circle(img,radius,center,elapsed_time,end_time):
    axes = (radius,radius)
    angle = 0
    startAngle = 0
    endAngle = 360/end_time*elapsed_time
    color = (255,255,255)
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color,5)
    return img

class TorqueTool:
    def __init__(self):
        pass
    def read_torque_values(self,base_cyclic):
        expected_number_of_actuators = 7 #works for 7dof Gen3
        feedback = base_cyclic.RefreshFeedback()
        torque_values=[]
        for axis in range(expected_number_of_actuators):
            torque_values.append(round(feedback.actuators[axis].torque,3))
            #print("the torque in axis %d is: "%axis+str(round(feedback.actuators[axis].torque,3))+" Nm") #showing torque of axis 1
        return torque_values

    def read_torque_values_computed(self,base_cyclic):
        feedback = base_cyclic.RefreshFeedback()
        tool_external_wrench=[
            round(feedback.base.tool_external_wrench_force_x,2),
            round(feedback.base.tool_external_wrench_force_y,2),
            round(feedback.base.tool_external_wrench_force_z,2),
            round(feedback.base.tool_external_wrench_torque_x,2),
            round(feedback.base.tool_external_wrench_torque_y,2),
            round(feedback.base.tool_external_wrench_torque_z,2),
            ]
        return tool_external_wrench

    def tap_toggle(self,base_cyclic,torque_threshold=3):
        init_torque_vals=self.read_torque_values(base_cyclic)
        tap=False
        print("Waiting for a physical tap on the robot...")
        while tap==False:
            current_torque_vals=self.read_torque_values(base_cyclic)
            #list_a=[1,2,3,-4,5,6]
            #list_b=[1,2,2,-4,5,6]
            torque_delta=list(map(sub,init_torque_vals,current_torque_vals))
            #print(delta)
            abs_torque=0
            #add all torque differences up to the resulting torque difference
            for elem in torque_delta:
                abs_torque+=abs(elem)
            if abs_torque>=torque_threshold:
                tap=True
            #print(abs_torque)
        return print("A torque bigger than %f Nm was registered (%f Nm)"%(torque_threshold,abs_torque))
    def tap_toggle2(self,base_cyclic,torque_threshold=3):
        print("Waiting for a physical tap on the robot...")
        init_torque=self.read_torque_values(base_cyclic)
        init_torque = [abs(val) for val in init_torque]
        tap=False
        while tap==False:
            try:
                current_torque = self.read_torque_values(base_cyclic)
                current_torque = [abs(val) for val in current_torque]
                delta_torque = [abs(round(val1-val2,3)) for val1,val2 in zip(current_torque,init_torque)]
                for val in delta_torque:
                    if val>torque_threshold:
                        tap=True
            except (KeyboardInterrupt,SystemExit):
                raise
        return print("A torque bigger than %f Nm was registered"%(torque_threshold))

    def tap_toggle3(self,e0,base_cyclic,f_t_thresholds=[0,0,5,0,0,0]):
        # use the computed force values of tool_external_wrench_force_...
        # should be the best approach
        # Works like this: When a force bigger than 5 Newtons is applied on z-axis, return
        print("Waiting for a physical tap on the robot...")
        # receive indices from input f_t_thresholds for the desired axis
        indices=[]
        for index, temp in enumerate(f_t_thresholds):
            if f_t_thresholds[index] != 0:
                indices.append(index)
        init_torque=self.read_torque_values_computed(base_cyclic)
        init_torque = [abs(val) for val in init_torque]
        tap=False
        while tap==False and e0.is_set()==False:
            try:
                current_torque = self.read_torque_values_computed(base_cyclic)
                current_torque = [abs(val) for val in current_torque]
                delta_torque = [abs(round(val1-val2,3)) for val1,val2 in zip(current_torque,init_torque)]
                for index in indices:
                    if delta_torque[index] >= f_t_thresholds[index]:
                        tap=True
                        print("A force/torque of %f N/Nm (above threshold) was registered on axis %d"%(delta_torque[index],index))
                time.sleep(0.05)
            except (KeyboardInterrupt,SystemExit):
                raise
        return

    def tap_toggle_single_axis(self,base_cyclic,torque_threshold=3,axis_id=6):
        init_torque_vals=self.read_torque_values(base_cyclic)
        tap=False
        print("Waiting for a physical tap on axis %d..."%(axis_id+1))
        while tap==False:
            current_torque_vals=self.read_torque_values(base_cyclic)
            #list_a=[1,2,3,-4,5,6]
            #list_b=[1,2,2,-4,5,6]
            torque_delta=list(map(sub,init_torque_vals,current_torque_vals))
            torque_delta=init_torque_vals[axis_id]-current_torque_vals[axis_id]
            #print(delta)
            # abs_torque=0
            #add all torque differences up to the resulting torque difference

            if abs(torque_delta)>=torque_threshold:
                tap=True
            #print(abs_torque)
        return print("A torque bigger than %f Nm was registered (%f Nm)"%(torque_threshold,abs(torque_delta)))



class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape
def detect_rectangle_middle(image): #detectiong contours
    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    resized = imutils.resize(image, width=400)
    ratio = image.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print(len(cnts))
    #debug_display(thresh)
    cnts = imutils.grab_contours(cnts)
    if len(cnts)>1:
        cnts=cnts[1:] #use not the outer contour
    else: #nothing found
        return image, (0,0), False
    sd = ShapeDetector()
    # loop over the contours
    for c in cnts:
        #c=cnts[0]
        #CONTINUE
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        try:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        except: #evade div/0 error if contour is too small/faulty
            return image, (0,0), False
        shape = sd.detect(c)
        if shape=="rectangle" and M['m00']>3000 and M['m00']<20000:
        #condition fitted to the size of the test rectangle
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image,(cX,cY),10,(255,255,0),1)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
            # show the output image
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)
            #return image
            return image, (cX,cY), True
    return image, (0,0), False

def rotate_to_target(delta,
    w,h,max_speed,min_speed,
    target_circle_size,base,base_cyclic,test_mode):
    # global log_string
    #linear speed adaption to target distance
    #for x
    distance_threshold1=0.05 #slow when this distance was reached (fraction of the image (e.g. 10%))
    distance_threshold2=0.5 #fast when this distance was reached (fraction of the image (e.g. 50%))
    if abs(delta[0])>w*distance_threshold1 and abs(delta[0])<w*distance_threshold2:
        m_x=(max_speed-min_speed)/(w*distance_threshold2-w*distance_threshold1)#Anstieg max 5degrees, min 1 degree
        deg_x=m_x*abs(delta[0])+min_speed-m_x*w*distance_threshold1
    elif abs(delta[0])<=w*distance_threshold1:
        deg_x=min_speed
    else:
        deg_x=max_speed

    if abs(delta[1])>h*distance_threshold1 and abs(delta[1])<h*distance_threshold2:
        m_y=(max_speed-min_speed)/(h*distance_threshold2-h*distance_threshold1)#Anstieg
        deg_y=m_y*abs(delta[1])+min_speed-m_y*h*distance_threshold1
    elif abs(delta[1])<=h*distance_threshold1:
        deg_y=min_speed
    else:
        deg_y=max_speed

    pose_increment=[0,0,0,0,0,0]
    pose_increment_joint=[0,0,0,0,0,0,0]
    threshold_=target_circle_size
    debug_str="Delta: "+str(delta)+" "
    ##BASE
    """if delta[0]>threshold_:
        pose_increment[5]=-deg_x #rot(z) base
    elif delta[0]<-threshold_:
        pose_increment[5]=deg_x #rot(z) base
    if delta[1]>threshold_:
        pose_increment[3]=deg_y #rot(x) base
    elif delta[1]<-threshold_:
        pose_increment[3]=-deg_y #rot(x) base"""
    ##TOOL
    if delta[0]>threshold_:
        pose_increment[4]=-deg_x #rot(y) tool
        debug_str+="rotating my tool right (-y) with "+str(-deg_x)+" deg/s"
    elif delta[0]<-threshold_:
        pose_increment[4]=deg_x #rot(y) tool
        debug_str+="rotating my tool left (+y) with "+str(deg_x)+" deg/s"
    if delta[1]>threshold_:
        pose_increment[3]=deg_y #rot(x) base
        debug_str+=" and rotating my tool down (+x) "+str(deg_y)+" deg/s"
    elif delta[1]<-threshold_:
        pose_increment[3]=-deg_y #rot(x) base
        debug_str+=" and rotating my tool up (-x) "+str(-deg_y)+" deg/s"
    #check if theta_y is level (this is only beta, works only well in two quadrants)
    if test_mode==False:
        current_pose=highlevel_movements.get_tcp_pose(base_cyclic)
        # print("current_pose[4]: ",current_pose[4])
        theta_y=current_pose[4]
        #theta_y=abs(theta_y)
        if theta_y<90: #perfekt: 10 / 170
            deg_z=6
        elif theta_y>90:
            deg_z=-6
        #theta_y=abs(theta_y)
        # if -180 <= current_pose[4] < -90:
        #     pose_increment[5]=deg_z
        # if 2 <= theta_y < 6:
        #     pose_increment[5]=-deg_z
        # elif 14 <= theta_y < 90:
        #     pose_increment[5]=deg_z
        # elif 90 <= theta_y < 168:
        #     pose_increment[5]=-deg_z
        # elif 174 <= theta_y <= 178:
        #     pose_increment[5]=deg_z
        if -90 <= theta_y < 4:
            pose_increment[5]=-deg_z
        elif 12 <= theta_y < 90:
            pose_increment[5]=+deg_z
        elif 90 <= theta_y < 168:
            pose_increment[5]=-deg_z
        elif 176 <= theta_y <= 180:
            pose_increment[5]=+deg_z
        elif -180 <= theta_y <= -90:
            pose_increment[5]=-deg_z
        #print("theta_y :",theta_y)
        # log_string+="rot dir: "+str(pose_increment[5])+" "
        # TODO für jeden einzelnen Quadranten die Regeln ermitteln
        #print(debug_str)
        #if time.time()-t0>: #Cannot send continuously
            #base.Stop()
        # print("pose_increment: ",pose_increment)
        highlevel_movements.example_twist_command(base,pose_increment)
    if pose_increment==[0,0,0,0,0,0]:
        #robot within target
        return True
    else:
        return False
