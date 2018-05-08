import sys
sys.path.append("facerecogniton/MTCNN")


from MTCNN.MtcnnDetector import MtcnnDetector
from MTCNN.detector import Detector
from MTCNN.fcn_detector import FcnDetector
from MTCNN.mtcnn_model import P_Net, R_Net, O_Net


class MTCNNDetect(object):
    def __init__(self, model_path = "models", threshold = [0.6, 0.7, 0.7], factor = 0.709, scale_factor = 2, minsize=20):
        epoch = [18, 14, 16]
        prefix = [model_path + '/PNet_landmark/PNet', model_path + '/RNet_landmark/RNet', model_path + '/ONet_landmark/ONet']
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        detectors = [None, None, None]
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet
        RNet = Detector(R_Net, 24, 1, model_path[1])
        detectors[1] = RNet
        ONet = Detector(O_Net, 48, 1, model_path[2])
        detectors[2] = ONet

        self.mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=minsize,
                                   stride=scale_factor, threshold=threshold, slide_window=False)
    def detect(self, img):
        rects, landmarks = self.mtcnn_detector.detect(img)
        
        newRects = []
        for rect in rects:
            newRect = (int(rect[0]), int(rect[1]), int(rect[2] - rect[0]), int(rect[3] - rect[1]))
            newRects.append(newRect)
        return newRects, landmarks
