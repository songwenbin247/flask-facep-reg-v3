import cv2
from base_camera import BaseCamera
import facerecogniton.facerecogniton as facerecg
import Queue
import numpy as np


class Camera(BaseCamera):
    video_source = range(facerecg.CAMERA_NUMBER)
    buffer_count = 10
    reg_ret = []

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def set_buffer_count(count):
        Camera.buffer_count = count

    @staticmethod
    def frames():
        framequeue = []
        cameras = []
#        for i in range(facerecg.CAMERA_NUMBER):
#            cameras.append(cv2.VideoCapture(Camera.video_source[i]))
#            if not cameras[i].isOpened():
#                raise RuntimeError('Could not start camera. Index:' , i)
        cameras.append(cv2.VideoCapture(0))
        cameras.append(cv2.VideoCapture(1))
        cameras.append(cv2.VideoCapture("rtsp://admin:a12345678@10.193.20.162/mpeg4/ch1/sub/av_stream"))
        framequeue = Queue.Queue(maxsize=Camera.buffer_count)

        #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        recg_ret = []


        for j in range(Camera.buffer_count - 1):
            images = []
            for i in range(facerecg.CAMERA_NUMBER):
                _, img =  cameras[i].read()
                images.append(img)
            framequeue.put(images)

        while True:
            images = []
            # read current frame
            for i in range(facerecg.CAMERA_NUMBER):
                _, img = cameras[i].read()
                images.append(img)

            facerecg.proCvFrame(images)
            framequeue.put(images)
            images = framequeue.get()

            rets = facerecg.getResult()
            print rets
            #if rets is not None:
            for (i, each) in  enumerate(rets):
                print("index", i)
                for ret in each:
                    #draw bounding box for the face
                    rect = ret['rect']
                    cv2.rectangle(images[i],(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(0,0,255),2)
                    cv2.putText(images[i], ret['name'],(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            # encode as a jpeg image and return it
            if (len(images) == 1):
                final = images[0]
            elif (len(images) == 2):
                final = np.concatenate((images[0], images[1]), axis=1)
            elif (len(images) == 3):
                final1 = np.concatenate((images[0], images[1]), axis=1)
                final2 = np.concatenate((images[2], images[2]), axis=1)
                final = np.concatenate((final1, final2), axis=0)
            yield cv2.imencode('.png', final)[1].tobytes()
