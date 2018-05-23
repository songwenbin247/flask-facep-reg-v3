import cv2
from base_camera import BaseCamera
import facerecogniton.facerecogniton as facerecg
import Queue
import numpy as np
import time,os

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


class Camera(BaseCamera):
    video_source = []
    camera_number = 0
    buffer_count = 5
    reg_ret = []

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source
        Camera.camera_number = len(source)
        if Camera.camera_number > 2:
            Camera.camera_number = 2

    @staticmethod
    def set_buffer_count(count):
        Camera.buffer_count = count

    @staticmethod
    def frames():
        framequeue = []
        cameras = []
        for i in range(Camera.camera_number):
            cameras.append(cv2.VideoCapture(Camera.video_source[i]))
            if not cameras[i].isOpened():
                raise RuntimeError('Could not start camera. Index:' , i)

            if(Camera.camera_number == 1):
                cameras[i].set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 800)
                cameras[i].set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 450)
            elif(Camera.camera_number == 2):
                cameras[i].set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
                cameras[i].set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        framequeue = Queue.Queue(maxsize=Camera.buffer_count)


        for j in range(Camera.buffer_count - 1):
            images = []
            for i in range(Camera.camera_number):
                _, img =  cameras[i].read()
                images.append(img)
            framequeue.put(images)

        last_time = 0
        last_rets = []

        while True:
            images = []
            # read current frame
            current = time.time() * 1000

            if current - last_time < 30:
                time.sleep(0.001)
                continue
            last_time = current

            for i in range(Camera.camera_number):
                _, img = cameras[i].read()
                #after = rotate(img, -90)
                #images.append(after)
                images.append(img)

            facerecg.proCvFrame(images)
            framequeue.put(images)
            images = framequeue.get()

            rets = facerecg.getResult()
            if rets is None:
                rets = last_rets
            else:
                last_rets = rets

            for (i, each) in  enumerate(rets):
                for ret in each:
                    #draw bounding box for the face
                    rect = ret['rect']
                   # cv2.rectangle(images[i],(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(0,0,255),2)
                   # cv2.putText(images[i], ret['name'],(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
		    cv2.rectangle(images[i],(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(127,255,0),1)
					
					#
                    # draw thicking corners
					
		    int_x=rect[2]/5
		    int_y=rect[3]/5
                    cv2.line(images[i],(rect[0],rect[1]),(rect[0] + int_x,rect[1]),(127,255,0),3)
                    cv2.line(images[i],(rect[0],rect[1]),(rect[0],rect[1]+int_y),(127,255,0),3)
                    cv2.line(images[i],(rect[0],rect[1]+int_y*4),(rect[0],rect[1]+rect[3]),(127,255,0),3)
                    cv2.line(images[i],(rect[0],rect[1]+rect[3]),(rect[0] + int_x,rect[1]+rect[3]),(127,255,0),3)
                    cv2.line(images[i],(rect[0]+ int_x*4,rect[1]+rect[3]),(rect[0] + rect[2],rect[1]+rect[3]),(127,255,0),3)
                    cv2.line(images[i],(rect[0] + rect[2],rect[1]+rect[3]),(rect[0] + rect[2],rect[1]+int_y*4),(127,255,0),3)
                    cv2.line(images[i],(rect[0] + rect[2],rect[1]+int_y),(rect[0] + rect[2],rect[1]),(127,255,0),3)
                    cv2.line(images[i],(rect[0] + int_x*4,rect[1]),(rect[0] + rect[2],rect[1]),(127,255,0),3)
		    #draw middle line
		    line_x=rect[2]/8
                    cv2.line(images[i],(rect[0]-line_x,rect[1]+rect[3]/2),(rect[0] + line_x,rect[1]+rect[3]/2),(127,255,0),1)
                    cv2.line(images[i],(rect[0]+rect[2]/2,rect[1]+rect[3]-line_x),(rect[0]+rect[2]/2,rect[1]+rect[3]+line_x),(127,255,0),1)
                    cv2.line(images[i],(rect[0]+line_x*7,rect[1]+rect[3]/2),(rect[0]+line_x*9,rect[1]+rect[3]/2),(127,255,0),1)
                    cv2.line(images[i],(rect[0]+rect[2]/2,rect[1]-line_x),(rect[0]+rect[2]/2,rect[1]+line_x),(127,255,0),1)
            	   #write name text
		   # cv2.putText(images[i], ret['name'],(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(100,255,255,255),2)
                    if(ret['name']!=None and ret['name']!= " " and ret['name']!= ""):
                        cv2.putText(images[i], ret['name'],(rect[0]+rect[2],rect[1]),cv2.FONT_HERSHEY_COMPLEX,int_y*1.0/40,(242,243,231),2)
                        cv2.putText(images[i], 'Employee',(rect[0]+rect[2],rect[1]+rect[3]/5),cv2.FONT_HERSHEY_COMPLEX,int_y*1.0/50,(222,223,215),1)
                        cv2.putText(images[i], 'Digital Networking',(rect[0]+rect[2],rect[1]+rect[3]/3),cv2.FONT_HERSHEY_COMPLEX,int_y*1.0/60,(214,215,206),1)


            if (len(images) == 1):
                final1 = images[0]
                final = cv2.copyMakeBorder(final1,22,23,232,232, cv2.BORDER_CONSTANT,value=[255,255,255])
            elif (len(images) == 2):
                final1 = np.concatenate((images[0], images[1]), axis=1)
                final = cv2.copyMakeBorder(final1,48,48,0,0, cv2.BORDER_CONSTANT,value=[255,255,255])
            elif (len(images) == 3):
                final1 = np.concatenate((images[0], images[1]), axis=1)
                final2 = np.concatenate((images[2], images[2]), axis=1)
                final = np.concatenate((final1, final2), axis=0)
            elif (len(images) == 4):
                final1 = np.concatenate((images[0], images[1]), axis=1)
                final2 = np.concatenate((images[2], images[3]), axis=1)
                final = np.concatenate((final1, final2), axis=0)
            yield cv2.imencode('.png', final)[1].tobytes()
