#coding: utf-8
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
import ctypes
import codecs


#FRGraph = FaceRecGraph();
aligner = AlignCustom();
extract_feature = FaceFeature()
face_detect = MTCNNDetect(scale_factor=3); #scale_factor, rescales image for faster detection
feature_data_set = None

LIB = None
FEATURE = ctypes.c_double *  128
POS = ctypes.c_int
NAME = ctypes.c_wchar_p

'''
Description:
Images from Video Capture -> detect faces' regions -> crop those faces and align them 
    -> each cropped face is categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Search for matching subjects in the dataset based on the types of face positions. 
    -> The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
    (Distance threshold is 0.6, percentage threshold is 70%)
    
'''


def recog_process_frame(frame):
    rects, landmarks = face_detect.detect_face(frame,40);#min face size is set to 80x80
    aligns = []
    positions = []
    rets = []
    for (i, rect) in enumerate(rects):
        aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
        aligns.append(aligned_face)
        positions.append(face_pos)
    if (len(aligns) == 0):
        return rets
    features_arr = extract_feature.get_features(aligns)
    recog_data = findPeople_Python(features_arr,positions);
    for (i,rect) in enumerate(rects):
        rets.append({"name":recog_data[i], "rect":rect, "pos":positions[i]})
    return rets

'''
facerec_128D.txt Data Structure:
{
"Person ID": {
    "Center": [[128D vector]],
    "Left": [[128D vector]],
    "Right": [[128D Vector]]
    }
}
This function basically does a simple linear search for 
^the 128D vector with the min distance to the 128D vector of the face on screen
'''

def findPeople_Python(features_arr, positions, thres = 0.5, percent_thres = 97):
    regRes = [];
    for (i,features_128D) in enumerate(features_arr):
        returnRes = " ";
        smallest = sys.maxsize
        for person in feature_data_set.keys():
            person_data = feature_data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    returnRes = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage > percent_thres :
            regRes.append(returnRes)
            #regRes.append(returnRes+"-"+str(round(percentage,1))+"%")
        else:
            regRes.append(" ")
    return regRes


def findPeople_Optee(features_arr, positions, thres = 0.5, percent_thres = 97):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    regRes = [];
    for (i,features_128D) in enumerate(features_arr):
        cinput1 = FEATURE()
        cpos = POS()
        for tmp in range(128):
            cinput1[tmp] = features_128D[tmp]

        if (positions[i] == "Right"):
            cpos = 0
        elif (positions[i] == "Left"):
            cpos = 1
        elif (positions[i] == "Center"):
            cpos = 2
        ret = CLIB.find_people(cinput1, cpos, ctypes.c_double(thres), ctypes.c_int(percent_thres))

        if ret is not None:
            regRes.append(ret.encode("utf-8"))
        else:
            regRes.append(" ".encode("utf-8"))
    return regRes

def detect_people(frame):
    rects, landmarks = face_detect.detect_face(frame,40);#min face size is set to 80x80
    rets = []
    for (i, rect) in enumerate(rects):
        rets.append({"name":"", "rect":rect, "pos":"None"})
    return rets

person_imgs = {"Left" : [], "Right": [], "Center": []}
person_name = ""
def train_start(name):
    global person_imgs,person_name
    person_name = name
    person_imgs = {"Left" : [], "Right": [], "Center": []}
    return True

def get_names_Python():
    names = []
    for name in feature_data_set:
        names.append(name)
    return names

def get_names_Optee():
    return ["No"]

def save_feature_Optee(name, person_features):
    right_input = FEATURE()
    left_input = FEATURE()
    front_input = FEATURE()
    cname = NAME(name)
    right_data = person_features["Right"][0];
    left_data = person_features["Left"][0];
    front_data = person_features["Center"][0];

    for index in range(128):
        right_input[index] = right_data[index]
        left_input[index] = left_data[index]
        front_input[index] = front_data[index]
    CLIB.save_feature(cname, right_input, left_input, front_input)

def save_feature_Python(name, person_features):
    feature_data_set[name] = person_features;
    f = codecs.open('./models/facerec_128D.txt', 'w', 'utf-8');
    f.write(json.dumps(feature_data_set))
    f.close()

def __training_thread_local(callback):
    print("Start training")
    person_features = {"Left" : [], "Right": [], "Center": []};
    for pos in person_imgs:
        person_features[pos] = [np.mean(extract_feature.get_features(
                                         person_imgs[pos]),axis=0).tolist()]
    save_feature(person_name, person_features)
    print("Stop training")
    callback()

def train_finish(callback):
    t = threading.Thread(target=__training_thread_local, args=(callback,))
    t.start()
    return True

def train_process_people(frame):
    rects, landmarks = face_detect.detect_face(frame, 80);
    rets = []
    if (len(rects) == 1):
        aligned_frame, face_pos = aligner.align(160,frame,landmarks[0]);
        person_imgs[face_pos].append(aligned_frame)
        rets.append({"name":"", "rect":rects[0], "pos":face_pos})
    return rets

def delete_name_Optee(name):
    cname = NAME(name)
    ret = CLIB.delete_name(cname)
    if ret == 1:
        return True
    else:
        return False

def delete_name_Python(name):
    if (feature_data_set is not None and name in feature_data_set):
        del feature_data_set[name]
        f = codecs.open('./models/facerec_128D.txt', 'w', 'utf-8')
        f.write(json.dumps(feature_data_set))
        f.close()
        return True
    else:
        return False

def load_modules_Python():
    global feature_data_set
    f = codecs.open('./models/facerec_128D.txt','r', 'utf-8');
    feature_data_set = json.loads(f.read());
    f.close()

def load_modules_Optee():
    global CLIB
    ll = ctypes.cdll.LoadLibrary
    CLIB = ll("./libfeature.so")
    CLIB.load_feature()
    CLIB.find_people.restype = ctypes.c_wchar_p
    CLIB.delete_name.restype = ctypes.c_int


findPeople = findPeople_Python
load_modules = load_modules_Python
delete_name = delete_name_Python
save_feature = save_feature_Python
get_names = get_names_Python

load_modules()
