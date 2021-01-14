import sys
import cv2
import numpy as np
import time
from Subfunctions.WebcamVideoStream import WebcamVideoStream
from Subfunctions import detect
from Subfunctions.button_detector import button_cv_final
from threading import Thread
from queue import Queue
import os
from PIL import Image
import platform
from operator import add,sub
import multiprocessing
import numpy as np
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
def main():
    frame_count=0
    framerate=0
    # Start Edge TPU
    default_model_dir = './models'
    default_model1 = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    # default_model1 = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    default_labels1 = 'coco_labels.txt'
    default_threshold=0.6
    enable_grid=True
    enable_multi_inference = True
    enable_inference = False
    divider = 1 # sets the number of image sections for better results with smaller objects
                # e.g. divider = 6 --> image is split into a grid of 6*6 images

    fontStyle = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    fontThickness = 2

    engine = DetectionEngine(os.path.join(default_model_dir,default_model1))
    labels = dataset_utils.read_label_file(os.path.join(default_model_dir,default_labels1)) if default_labels1 else None

    def spin_up_engine(input):
        detected_labels=[]
        if len(input) == 6: # added a rectangle paarmeter for multi-segment-inferencing
            engine,labels,pil_im,default_threshold,exclude_list,rectangle = input
            rectangle = np.array([rectangle[0],rectangle[0]])
            # where rectangle is a part of the image [pt1,pt2]
        else: #for usage without multi-segment-inferencing
            engine,labels,pil_im,default_threshold,exclude_list = input
            rectangle = np.array([[0,0],[0,0]])
        objs = engine.detect_with_image(pil_im,
                                        threshold=default_threshold,
                                        keep_aspect_ratio='store_true',
                                        relative_coord=False,
                                        top_k=99)
        if objs:
            for obj in objs:
                obj_label=labels[obj.label_id]
                # print(obj.bounding_box)
                obj.bounding_box += rectangle # convert bbox position from image segment into real image position
                # print("coverted bbox: ",obj.bounding_box)
                x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                if x1-x0>w/2 or obj_label in exclude_list:
                    objs.remove(obj)
                    continue #the bounding box is too big, must be false-positive
                else:
                    detected_labels.append(obj_label)
        return objs,labels,detected_labels
    que = Queue()


    #cap = WebcamVideoStream(0).start()
    cap = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
    cap.set(3, 1280)
    cap.set(4, 720)


    time.sleep(0.5)
    # cv2_im = cv2.imread("./models/crowd.jfif")
    w = int(cap.get(3))
    h = int(cap.get(4))
    print(h, "/", w)
    # out = cv2.VideoWriter('./detection_%s.avi'%(datetime.today().strftime('%Y-%m-%d-%H:%M:%S')),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
    # out = cv2.VideoWriter('./detection_.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
    

    t0=time.time()
    while True:
        img_segments=[[[0,0],[w,h]]]
        ret, cv2_im = cap.read()
        #cv2_im = cv2.rotate(cv2_im, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

        
        # cv2_im = cv2.imread("./models/kites.png")
        # cv2_im = cv2.imread("./models/crowd.jfif")
        (h, w) = cv2_im.shape[:2]
        x_step = int(w/divider)
        y_step = int(h/divider)
        # outer
        for x in range(divider):
            for y in range(divider):
                pt1 = x*x_step,y*y_step
                pt2 = (x+1)*x_step,(y+1)*y_step
                if enable_grid==True: cv2.rectangle(cv2_im,pt1,pt2,(0,0,255),4)
                if enable_multi_inference==True: img_segments.append([pt1,pt2])
        img_segments.append([pt1,pt2])
        # middle 1
        pt1 = int(w/2-0.5*x_step),int(h/2-0.5*y_step)
        pt2 = int(w/2+0.5*x_step),int(h/2+0.5*y_step)
        if enable_grid==True: cv2.rectangle(cv2_im,pt1,pt2,(255,0,0),2)
        if enable_multi_inference==True: img_segments.append([pt1,pt2])
        # middle 2
        pt1 = int(w/2-x_step),int(h/2-y_step)
        pt2 = int(w/2+x_step),int(h/2+y_step)
        if enable_grid==True: cv2.rectangle(cv2_im,pt1,pt2,(255,255,0),2)
        if enable_multi_inference==True: img_segments.append([pt1,pt2])
        #convert to PIL
        pil_im = Image.fromarray(cv2_im.copy())
        #Start the threads for each image segment
        thread_list=[]
        if enable_inference == True:
            for rectangle in img_segments:
                left = rectangle[0][0] #x1
                top = rectangle[0][1] #y1
                right = rectangle[1][0] #x2
                bottom = rectangle[1][1] #y2
                # print("crop: ",(left, top, right, bottom))

                pil_im_segment = pil_im.crop((left, top, right, bottom))

                t1=Thread(
                    target=lambda q,
                    arg1: q.put(spin_up_engine(arg1)),
                    args=(que,[engine,labels,pil_im_segment,default_threshold,[],rectangle])
                    )
                # thread_list.append(t1)
                t1.start()
                t1.join()
                # objs = engine.detect_with_image(pil_im_segment,
                #                                 threshold=default_threshold,
                #                                 keep_aspect_ratio='store_true',
                #                                 relative_coord=False,
                #                                 top_k=10)
                # cv2_im = detect.append_objs_to_img(cv2_im, objs, labels)
            # for thread in thread_list:
            #     thread.join()
            # Check thread's return value
            detected_labels_all=[]
            # # sys.exit()
            #
            while not que.empty():
                #Retreive from threding queue
                objs,labels,detected_labels = que.get()
                #button/Contour detector
                # cv2_im, __ = button_cv_final.find_buttons_in_bbox(cv2_im, objs)
                # draw objects
                cv2_im = detect.append_objs_to_img(cv2_im, objs, labels,bbox_only=False)
                # write_to_temporary_dict(objs) # write to temporary dict
                detected_labels_all+=detected_labels
        t1=time.time()
        frame_count+=1
        if t1-t0>=1.0:
            framerate=frame_count
            frame_count=0
            t0=time.time()
        cv2.putText(cv2_im, "Framerate: "+str(framerate)+" Framecount: "+str(frame_count), (10,15),fontStyle, 0.5, (255,255,255), 2)

        cv2.imshow("Camera Feed", cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # cap.stop()
    cap.release()
    # out.release()
    cv2.destroyWindow(str(0))

if __name__ == '__main__':
    main()
