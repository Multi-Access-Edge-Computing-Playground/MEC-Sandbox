import multiprocessing
import cv2
from imutils.video import WebcamVideoStream
import time
caps=[1,0,'rtsp://192.168.1.10/color']

def cam(cap_id,e):
    cap = WebcamVideoStream(cap_id).start()
    while True:
        frame = cap.read()
        if e.is_set():
            cv2.circle(frame,(50,50),10,(255,0,255),2)
        cv2.imshow(str(cap_id), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.stop()
    cv2.destroyWindow(str(cap_id))
    return

if __name__ == '__main__':
    jobs = []
    e = multiprocessing.Event()
    for cap_id in caps:
        p = multiprocessing.Process(target=cam, args=(cap_id,e))
        jobs.append(p)
        p.start()
    for i1 in range(len(caps)):
        for i0 in range(3):
            time.sleep(1)
            print("waited %d seconds"%i0)
        e.set()
        #jobs[i1].terminate()
        #jobs[i1].join()
