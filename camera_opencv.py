import cv2
from base_camera import BaseCamera
import facerecogniton.facerecogniton as facerecg
import Queue


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
        images = []
        for i in range(facerecg.CAMERA_NUMBER)
            cameras.append(cv2.VideoCapture(Camera.video_source[i]))
            if not cameras[i].isOpened():
                raise RuntimeError('Could not start camera. Index:' , i)
        framequeue = Queue.Queue(maxsize=Camera.buffer_count)

        #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        recg_ret = []


        for i in range(facerecg.CAMERA_NUMBER)
            for j in range(Camera.buffer_count - 1):
                _, img =  cameras[i].read()
                images.append(img)
            framequeue.put(img)

        while True:
            images = []
            # read current frame
            for i in range(facerecg.CAMERA_NUMBER):
                _, img = cameras[i].read()
                images_recg.append(img)

            facerecg.proCvFrame(images)
            framequeue.put(images)
            images = framequeues.get()

            rets = facerecg.getResult()

            for i in range(facerecg.CAMERA_NUMBER):
                for ret in rets[i]:
                    #draw bounding box for the face
                    rect = ret['rect']
                    cv2.rectangle(images[i],(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(0,0,255),2)
                    cv2.putText(images[i], ret['name'],(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            # encode as a jpeg image and return it
            if (len(images) == 1):
                final = images[0]
            else:
                print("Error")
            yield cv2.imencode('.png', final)[1].tobytes()
