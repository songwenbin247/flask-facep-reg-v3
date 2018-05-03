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

def training_proframe(name, aligned_frame, pos):
    global person_images,feature_data_set
    person_images[name][pos].append(aligned_frame)
    return True

def training_proframe_detect(name, frame):
    print "training_proframe_detect", name
    rects, landmarks = face_detect.detect_face(frame, 20);
    print rects,landmarks
    if (len(rects) == 1):
        aligned_frame, face_pos = aligner.align(160,frame,landmarks[0]);
        person_images[name][face_pos].append(aligned_frame)
    return True

def get_train_status(name):
    if (name in feature_data_set):
        return "FINISHED"
    elif (name in person_images):
        return "TRAINING"
    else:
        return "NOEXIS"

def get_images_num(name):
    print "get_images_num", name
    if (name not in person_images and name not in feature_data_set):
        return {'l':0, 'r':0, 'c':0}
    elif (name not in person_images and name in feature_data_set):
        return {'l':5, 'r':5, 'c':10}
    else:
        return {'l':len(person_images[name]["Left"]), 'r':len(person_images[name]["Right"]),
                'c':len(person_images[name]["Center"])}

def __training_thread(name, callback):
    print "__training_thread"
    global person_images,feature_data_set
    person_features = {"Left" : [], "Right": [], "Center": []};
    for pos in person_images[name]:
        person_features[pos] = [np.mean(extract_feature.get_features(
                                         person_images[name][pos]),axis=0).tolist()]
    if (feature_data_set is not None):
        feature_data_set[name] = person_features;
        f = codecs.open('./models/facerec_128D.txt', 'w', 'utf-8');
        f.write(json.dumps(feature_data_set))
        f.close()
        del person_images[name]
    if callback:
        callback(name)

def training_finish(name, callback=None):
    print "training_finish"
    t = threading.Thread(target=__training_thread, args=(name, callback,))
    t.start()
    return True

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
