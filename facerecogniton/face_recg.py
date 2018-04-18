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
import codecs
import requests


#FRGraph = FaceRecGraph();
aligner = AlignCustom();
extract_feature = FaceFeature()
face_detect = MTCNNDetect(scale_factor=3); #scale_factor, rescales image for faster detection
feature_data_set = None

CAMERA_NUMBER = 4

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
    def __init__(self):
        self.aligns = []
        self.positions = []
        self.faces = []
        self.rets = []
        self.rects = []

def recog_process_frame(frames):
    cameras = []
    rets = []
    aligns = []
    positions = []

    for (index,frame) in enumerate(frames):
        cameras.append(CameraRouad())
        cameras[index].rects, landmarks = face_detect.detect_face(frame,20);#min face size is set to 80x80
        face_tracker[index].increase_frame()
        for (i, rect) in enumerate(cameras[index].rects):
            aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
            face = face_tracker[index].get_face_by_position(rect, aligned_face)
            cameras[index].faces.append(face)
            if (1):
#            if (face.get_name() == None):
#            if (face.get_name() == None and face.unknow_count % 2 == 0):
                cameras[index].aligns.append(aligned_face)
                cameras[index].positions.append(face_pos)
#            elif (face.get_name() == None and face.unknow_count % 2 != 0):
#                cameras[index].rets.append({"name":" ", "rect":rect})
            else:
                cameras[index].rets.append({"name":face.get_name(), "rect":rect})

    for c in cameras:
        aligns += c.aligns
        positions += c.positions
    if (len(aligns) == 0):
        for (index,camera) in enumerate(cameras):
            rets.append(camera.rets)
        return rets

    features_arr = extract_feature.get_features(aligns)
    recog_data = findPeople(features_arr,positions);

    j = 0
    for (index,camera) in enumerate(cameras):
        for (i,rect) in enumerate(camera.rects):
            face = camera.faces[i]
            if (1):
            #if (face.get_name() == None and face.unknow_count % 2 == 0):
                face.set_name(recog_data[j])
                camera.rets.append({"name":recog_data[j], "rect":rect})
                j += 1
        face_tracker[index].drop_timeout_face()
        rets.append(camera.rets)
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

def findPeople(features_arr, positions, thres = 0.6, percent_thres = 97):
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


def detect_people(frames):
    rets = []
    for frame in frames:
        rects, landmarks = face_detect.detect_face(frame,20);#min face size is set to 80x80
        ret_per_frame = []
        for (i, rect) in enumerate(rects):
            ret_per_frame.append({"name":"", "rect":rect, "pos":"None"})
        rets.append(ret_per_frame)
    return rets

person_imgs = {"Left" : [], "Right": [], "Center": []}
person_name = ""
def train_start_Local(name):
    global person_imgs,person_name,orgin_imgs
    person_name = name
    person_imgs = {"Left" : [], "Right": [], "Center": []}
    orgin_imgs = []
    return True

def train_start_Server(name):
    global person_imgs,person_name
    args = {'id': name}
    headers = {"Content-type":"application/json","Accept": "application/json"}
    r = requests.put(url, params=args, headers=headers)
    ret = json.loads(r.text)
    if ('state'in ret and ret['state'] == 'SUCCESS'):
        person_name = name
        person_imgs = {"Left" : [], "Right": [], "Center": []}
        orgin_imgs = []
        return True
    else:
        return False

def train_finish_Local(name):
    global person_imgs,person_name
    person_name = ""
    person_imgs = {"Left" : [], "Right": [], "Center": []}
    return True, get_names()

def train_finish_Server(name):
    global person_imgs,person_name
    print("train_finish_Server")
    if person_name == name:
        person_name = ""
        person_imgs = {"Left" : [], "Right": [], "Center": []}
        isme = True
    else:
        isme = False
    load_modules_Server()
    return isme, get_names() 

def get_names():
    print("get_names")
    names = []
    for name in feature_data_set:
        names.append(name)
    return names

def save_feature_Local(name, person_features):
    feature_data_set[name] = person_features;
    f = codecs.open('./models/facerec_128D.txt', 'w', 'utf-8');
    f.write(json.dumps(feature_data_set))
    f.close()

def __training_thread_local(callback):
    print("__training_thread_local")
    person_features = {"Left" : [], "Right": [], "Center": []};
    for pos in person_imgs:
        person_features[pos] = [np.mean(extract_feature.get_features(
                                         person_imgs[pos]),axis=0).tolist()]
    save_feature_Local(person_name, person_features)
    print("Stop training")
    callback()

def recod_finish_Local(callback):
    print "recod_finish_Local"
    t = threading.Thread(target=__training_thread_local, args=(callback,))
    t.start()
    return True

def __training_thread_server(callback):
    print "__training_thread"
    files = {}

    for pos in ["Center", "Left", "Right"]:
        for i,frame in enumerate(person_imgs[pos]):
            picf = StringIO.StringIO()
            pi = Image.fromarray(frame)
            pi.save(picf, format = "jpeg")
            picf.seek(0)
            files['file{}{}'.format(pos,i)] = ('{}{}.jpeg'.format(pos,i), picf, 'image/jpeg')

    args = {'id': person_name, 'end':'true'}
    headers = {"Content-type":"application/json","Accept": "application/json"}

    while(1):
        r = requests.post(url, params=args, files=files)
        ret = json.loads(r.text)
        if ('state' in ret and ret['state'] != 'FAILED'):
            print("break loop")
            break

    callback()

def recod_finish_Server(callback):
    print "recod_finish_Server"
    t = threading.Thread(target=__training_thread_server, args=(callback,))
    t.start()
    return True

def train_process_people(frames):
    frame = frames[0]
    rects, landmarks = face_detect.detect_face(frame, 20);
    ret_per_frame = []
    rets = []
    if (len(rects) == 1):
        aligned_frame, face_pos = aligner.align(160,frame,landmarks[0]);
        if (len(person_imgs[face_pos]) < 15):
            person_imgs[face_pos].append(aligned_frame)
        ret_per_frame.append({"name":"", "rect":rects[0], "pos":face_pos})
    rets.append(ret_per_frame)
    return rets

def delete_name_Server(name):
    if (feature_data_set is None or name not in feature_data_set):
        return False
    args = {'id': name}
    headers = {"Content-type":"application/json","Accept": "application/json"}
    r = requests.delete(url, params=args, headers=headers)
    ret = json.loads(r.text)
    if ('state'in ret and ret['state'] == 'SUCCESS'):
        return True
    else:
        return False

def delete_name_Local(name):
    if (feature_data_set is not None and name in feature_data_set):
        del feature_data_set[name]
        f = codecs.open('./models/facerec_128D.txt', 'w', 'utf-8')
        f.write(json.dumps(feature_data_set))
        f.close()
        return True
    else:
        return False

def load_modules_Server():
    print "Downloading modules from cloud."
    global feature_data_set, info_data_set
    headers = {"Content-type":"application/json","Accept": "application/json"}
    r = requests.get(url, headers=headers)
    if (r.status_code == 200):
        f = open('./models/facerec_128D.txt','w')
        f.write(r.content)
        f.close()
        feature_data_set = json.loads(r.content)

        return True
    else:
        print "Modules updated failed"
        return False

def load_modules_Local():
    global feature_data_set
    f = codecs.open('./models/facerec_128D.txt','r', 'utf-8');
    feature_data_set = json.loads(f.read());
    f.close()

load_modules = load_modules_Local
delete_name = delete_name_Local
train_start = train_start_Local
recod_finish = recod_finish_Local
train_finish = train_finish_Local
url = ''

def init_engine(serverip):
    global load_modules,delete_name,train_start,train_finish,recod_finish,url
    if (serverip == None):
        pass
    else:
        url  = 'http://{}:8383/train'.format(serverip)
        load_modules = load_modules_Server
        delete_name = delete_name_Server
        train_start = train_start_Server
        recod_finish = recod_finish_Server
        train_finish = train_finish_Server

    load_modules()
