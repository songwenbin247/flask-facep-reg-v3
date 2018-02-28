'''
Main program
@Author: David Vu

To execute simply run:
main.py

To input new user:
main.py --mode "input"

'''

import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
import argparse
import sys, time
import json
import numpy as np
from PIL import Image
import StringIO
import threading
import codecs


aligner = AlignCustom();
extract_feature = FaceFeature()
face_detect = MTCNNDetect(scale_factor=2); #scale_factor, rescales image for faster detection
feature_data_set = {}
person_images = {}



def training_start(name):
    global person_images,feature_data_set
    if (name in feature_data_set or name in person_images):
        return False

    person_images[name] = {"Left" : [], "Right": [], "Center": []}
    return True

def training_proframe(name, frame):
    global person_images,feature_data_set
    rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80
    for (i, rect) in enumerate(rects):
        aligned_frame, pos = aligner.align(160,frame,landmarks[i]);
        person_images[name][pos].append(aligned_frame)
    for (i,rect) in enumerate(rects):
        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0))
    return frame

def __training_thread(name, callback):
    global person_images,feature_data_set
    person_features = {"Left" : [], "Right": [], "Center": []};
    for pos in person_images[name]:
        person_features[pos] = [np.mean(extract_feature.get_features(
                                         person_images[name][pos]),axis=0).tolist()]
    if (feature_data_set is not None):
        del person_images[name]
        feature_data_set[name] = person_features;
        f = codecs.open('./models/facerec_128D.txt', 'w', 'utf-8');
        f.write(json.dumps(feature_data_set))
        f.close()
    if callback:
        callback()

def training_finish(name, callback=None):
    t = threading.Thread(target=__training_thread, args=(name, callback,))
    t.start()
    return True

def get_person_names():
    names = []
    for name in feature_data_set:
        names.append(name)
    return names

def delete_module(name):
    if (feature_data_set is not None and name in feature_data_set):
        del feature_data_set[name];
        f = codecs.open('./models/facerec_128D.txt', 'w', 'utf-8');
        f.write(json.dumps(feature_data_set))
        f.close()
        return True
    else:
        return False

def load_modules():
    global feature_data_set
    f = codecs.open('./models/facerec_128D.txt','r', 'utf-8');
    feature_data_set = json.loads(f.read());
    f.close()

load_modules()
