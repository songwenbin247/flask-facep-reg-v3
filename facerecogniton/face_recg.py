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
from face_tracker import *

#FRGraph = FaceRecGraph();
aligner = AlignCustom();
extract_feature = FaceFeature()
face_detect = MTCNNDetect(scale_factor=3); #scale_factor, rescales image for faster detection

CAMERA_NUMBER = 1

face_tracker = []
for i in range(CAMERA_NUMBER):
    face_tracker.append(FaceTracker())

'''
Description:
Images from Video Capture -> detect faces' regions -> crop those faces and align them 
    -> each cropped face is categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Search for matching subjects in the dataset based on the types of face positions. 
    -> The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
    (Distance threshold is 0.6, percentage threshold is 70%)
    
'''
class CameraRouad:
    def __init__();
        self.aligns = []
        self.positions = []
        self.faces = []
        self.rets = []
        self.rects = []

def recog_process_frame(frames)
    cameras = []
    rets = []

    for (index,frame) in enumerate(frames):
        cameras.append(CameraRouad())
        cameras[index].rects, landmarks = face_detect.detect_face(frame,40);#min face size is set to 80x80
        face_tracker[index].increase_frame()
        for (i, rect) in enumerate(cameras[index].rects):
            aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
            face = face_tracker.get_face_by_position(rect, aligned_face)
            faces.append(face)
            if (face.get_name() == None):
                cameras[index].aligns.append(aligned_face)
                cameras[index].positions.append(face_pos)
            else:
                cameras[index].rets.append({"name":face.get_name(), "rect":rect})

    if (len(aligns) == 0)
        return rets
    features_arr = extract_feature.get_features(aligns)
    recog_data = findPeople(features_arr,positions);

    for (index,camera) in enumerate(cameras):
        j = 0
        for (i,rect) in enumerate(rects):
            face = camera.faces[i]
            if (face.get_name() == None):
                face.set_name(recog_data[j])
                camera.rets.append({"name":recog_data[j], "rect":rect})
                j += 1
        face_tracker[index].drop_timeout_face()
        rets.append[camera.rets]
    return rets
  except Exception as e:
    print e

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
import ctypes
ll = ctypes.cdll.LoadLibrary
lib = ll("./libfeature.so")
lib.load_feature()
lib.find_people.restype = ctypes.c_wchar_p
lib.delete_name.restype = ctypes.c_int

FEATURE = ctypes.c_double *  128
POS = ctypes.c_int
NAME = ctypes.c_wchar_p


def findPeople(features_arr, positions, thres = 0.6, percent_thres = 95):
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
        ret = lib.find_people(cinput1, cpos, ctypes.c_double(thres), ctypes.c_int(percent_thres))

        if ret is not None:
            regRes.append(ret.encode("utf-8"))
        else:
            regRes.append(" ".encode("utf-8"))
    return regRes

def detect_people(frames):
    rets = []
    for frame in frames:
        rects, landmarks = face_detect.detect_face(frame,40);#min face size is set to 80x80
        ret_per_frame = []
        for (i, rect) in enumerate(rects):
            ret_per_frame.append({"name":"", "rect":rect, "pos":"None"})
        rets.append(ret_per_frame)
    return rets

person_imgs = {"Left" : [], "Right": [], "Center": []}
person_name = ""
def train_start(name):
    global person_imgs,person_name
    person_name = name
    person_imgs = {"Left" : [], "Right": [], "Center": []}
    return True

def __training_thread_local(callback):
    print("Start training")
    person_features = {"Left" : [], "Right": [], "Center": []};
    for pos in person_imgs:
        person_features[pos] = [np.mean(extract_feature.get_features(
                                         person_imgs[pos]),axis=0).tolist()]
    right_input = FEATURE()
    left_input = FEATURE()
    front_input = FEATURE()
    name = NAME(person_name)
    right_data = person_features["Right"][0];
    left_data = person_features["Left"][0];
    front_data = person_features["Center"][0];

    for index in range(128):
        right_input[index] = right_data[index]
        left_input[index] = left_data[index]
        front_input[index] = front_data[index]
    lib.save_feature(name, right_input, left_input, front_input)
    print("Stop training")
    callback()

def train_finish(callback):
    t = threading.Thread(target=__training_thread_local, args=(callback,))
    t.start()
    return True

def train_process_people(frames):
    frame = frames[0]
    rects, landmarks = face_detect.detect_face(frame, 80);
    ret_per_frame = []
    rets = []
    if (len(rects) == 1):
        aligned_frame, face_pos = aligner.align(160,frame,landmarks[0]);
        person_imgs[face_pos].append(aligned_frame)
        ret_per_frame.append({"name":"", "rect":rects[0], "pos":face_pos})
    rets.append(ret_per_frame)
    return rets

def delete_name(name):
    cname = NAME(name)
    ret = lib.delete_name(cname)
    if ret == 1:
        return True
    else:
        return False
