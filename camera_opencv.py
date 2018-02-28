import cv2
from base_camera import BaseCamera
import facerecogniton.facerecogniton as facerecg
import Queue


class Camera(BaseCamera):
    video_source = 0
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
        frameq = Queue.Queue(maxsize=Camera.buffer_count)
        camera = cv2.VideoCapture(Camera.video_source)
        #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        recg_ret = []

        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        for i in range(Camera.buffer_count - 1):
            _, img = camera.read()
            frameq.put(img)

        while True:
            # read current frame
            _, img = camera.read()
            facerecg.proCvFrame(img)
            frameq.put(img)
            img = frameq.get()

            r = facerecg.getResult()
            if r is not None:
                recg_ret = r

            for ret in recg_ret:
                #draw bounding box for the face
                rect = ret['rect']
                cv2.rectangle(img,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(0,0,255),2)
                cv2.putText(img, ret['name'],(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            # encode as a jpeg image and return it
            yield cv2.imencode('.png', img)[1].tobytes()
