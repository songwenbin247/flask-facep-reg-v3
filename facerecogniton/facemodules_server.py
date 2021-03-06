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

def __training_thread(name, callback):
    print "__training_thread"
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
