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
import hot_pool
from data_base import  FeaturesDB

from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import OneClassSVM
from sklearn import neighbors, datasets
from sklearn.neighbors import LocalOutlierFactor
from sklearn import neighbors, datasets
import ntpath
from scipy.spatial import distance

from face_meta import *
from hot_pool import Filter
from hot_pool import Anormal
from hot_pool import canvas_show


filter = Filter(3)
hot_faces= hot_pool.HotFaces(10)
#FRGraph = FaceRecGraph();
aligner = AlignCustom();
anormal = Anormal()
extract_feature = FaceFeature()
face_detect = MTCNNDetect(scale_factor=3); #scale_factor, rescales image for faster detection
feature_data_set = dict()

## Inter-subject and Intra-subject mean and variance
mean_intra = 0.5
var_intra  = 0.01
N_intra = 0
mean_inter = 1.5
var_inter  = 0.01
N_inter = 0
gaus_th = 0

gaus_pdf_ratio = 5
gaus_rhs = 0
svm_nu =0.02
svm_gamma = 5
svm_kernel= "rbf"

#perfmon variables
total_cnt = 1
false_pos_cnt = 0
false_neg_cnt = 0

personal_meta = dict()

LOF_model = None
KNN_model = None
vec_global = None
labels_global=None
CAMERA_NUMBER = 4




canvas_disc = canvas_show(60, 0, 1.5)

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
    landms= []
    face_locations = []

    for (index,frame) in enumerate(frames):
        cameras.append(CameraRouad())
        cameras[index].rects, cameras[index].landmarks = face_detect.detect_face(frame,40);#min face size is set to 80x80
        #if landmarks != []:
            # canvas.up_va(landmarks[0][0])
            # pis = filter.get_va(landmarks[0][0], landmarks[0][1])
            # canvas_filter.up_va(pis + 300)
            # if pis > 3:
            #     canvas_sw.up_va(400)
            # else:
            #     canvas_sw.up_va(300)
        face_tracker[index].increase_frame()
        for (i, rect) in enumerate(cameras[index].rects):
            aligned_face, face_pos = aligner.align(160,frame,cameras[index].landmarks[i])
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
        landms += c.landmarks.tolist()
        if(len(face_locations)==0):
            face_locations = np.array(c.rects)
        else:
            face_locations = np.concatenate((face_locations, np.array(c.rects)), axis=0)

    if (len(aligns) == 0):
        for (index,camera) in enumerate(cameras):
            rets.append(camera.rets)
        return rets

    features_arr = extract_feature.get_features(aligns)
    recog_data = findPeople(face_locations, features_arr,  landms, positions);


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

def train_LOF():
    global KNN_model
    global vec_global, labels_global

    labels = np.array([])
    vec = np.array([])

    if feature_data_set == None:
        return

    for id in feature_data_set.keys():
        # estimate intra-subject distribution
        centroid = personal_meta[id].centroid
        labels = np.append(labels, id)

        if (len(vec) == 0):
            vec = np.array([centroid])
        else:
            vec = np.concatenate((vec, np.array([centroid])), axis=0)


    print('vecs=%d, labels=%d' %(len(vec), len(labels)))
    #LOF_model = LocalOutlierFactor(n_neighbors=20)
    #LOF_model = LOF_model.fit(vec, labels)
    KNN_model = neighbors.KNeighborsClassifier(1, weights='uniform')
    KNN_model = KNN_model.fit(vec, labels)

    vec_global = vec
    labels_global = labels.tolist()


n = 0
mean_my = 0
def findPeople(face_locations, features_arr, landmarks ,positions):
    global  mean_intra, var_intra, N_intra, mean_inter, var_inter, N_inter
    global gaus_th, gaus_th0, gaus_rhs
    global total_cnt, false_pos_cnt, false_neg_cnt
    global personal_meta
    global  n, mean_my


    regRes = []
    for (i,features_128D) in enumerate(features_arr):
        returnRes = " "

        if feature_data_set == None:
            regRes.append("Unknown")
            return regRes
        #

        #print(KNN_model.predict_proba([features_128D]))
        (dist, ind) = KNN_model.kneighbors([features_128D], n_neighbors=1, return_distance=True)
        knn_label = KNN_model.predict([features_128D])[0]


        meta = personal_meta[knn_label]

        canvas_disc.up_va(dist)

        (isHot, isFull, name) = hot_faces.isHot(knn_label, landmarks[i])
        know = ""
        if isHot:
            regRes.append(name)
            know = name
            #print "person is hot"
            if not isFull:
                hot_faces.update(name, landmarks[i])
        else:
            if (dist < 0.35): #(meta.mean - 0*meta.var)):
                    print('person is going to hot')
                    hot_faces.update(knn_label, landmarks[i])
                    regRes.append(knn_label)
                    know = knn_label
            else:
                    regRes.append("Unknown")
                    know = "Unknown"
        print "know = %s name = %s mean/var/va = %f %f %f disc = %f " % (know, knn_label, meta.mean, meta.var, meta.mean - 0*meta.var, dist)
    return regRes


def estimate_feature_dist():
    global  mean_intra, var_intra, N_intra, mean_inter, var_inter, N_inter
    global gaus_rhs, gaus_th, gaus_th0, gaus_rhs, gaus_pdf_ratio
    global personal_metadata

    for id in feature_data_set.keys():
        personal_meta[id] = FaceMeta()
        vecs = []
        for pos in ['Left','Center', 'Right']:
            vecs += feature_data_set[id][pos]
        vecs = np.array(vecs)
        centroid = np.mean(vecs, axis=0)

        personal_meta[id].centroid = centroid
        dists = np.sqrt(np.sum(np.square(vecs - centroid), axis=1))
        personal_meta[id].mean = np.mean(dists)
        personal_meta[id].var = np.std(dists)

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
    if( feature_data_set != None):
        for name in feature_data_set:
            names.append(name)
    return names

def update_feature_dist(name, personal_features):
    estimate_feature_dist()



def save_feature_Local(name, person_features):

    #update estimate using existing dataset
    feature_data_set[name] = person_features;
    update_feature_dist(name, person_features);

    #train the classifier using updated features
    train_LOF();

    fdb = FeaturesDB()
    for p in ['Left', 'Right', 'Center']:
        fdb.add_features(name, person_features[p], p)
    total_cnt = 0
    false_neg_cnt = 0


def __training_thread_local(callback):
    print("__training_thread_local")
    person_features = {"Left" : [], "Right": [], "Center": []};
    for pos in person_imgs:
        #person_features[pos] = [np.mean(extract_feature.get_features( person_imgs[pos]),axis=0).tolist()]
        list_= extract_feature.get_features(person_imgs[pos]).tolist()
        person_features[pos] =  anormal.filter(list_).tolist()

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
            break

    callback()

def recod_finish_Server(callback):
    print "recod_finish_Server"
    t = threading.Thread(target=__training_thread_server, args=(callback,))
    t.start()
    return True
import time
def train_process_people(frames):
    frame = frames[0]
    rects, landmarks = face_detect.detect_face(frame, 20);
    ret_per_frame = []
    rets = []
    if (len(rects) == 1):
        pis = filter.get_va(landmarks[0][0], landmarks[0][1])
        if pis > 3:
            aligned_frame, face_pos = aligner.align(160,frame,landmarks[0]);
            #if (len(person_imgs[face_pos]) < 20):
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
        fdb = FeaturesDB()
        fdb.del_person(name)

        estimate_feature_dist();
        train_LOF();

        for id in personal_meta.keys():
            print('id=%s, m=%f, var=%f, sigma=%f' %(id, personal_meta[id].mean, personal_meta[id].var, np.sqrt(personal_meta[id].var )))

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
    global personal_meta
    fdb = FeaturesDB()
    names = fdb.get_names()
    for name in names:
        fl = [x for x in fdb.features(name, 'Left')]
        fr = [x for x in fdb.features(name, 'Right')]
        fc = [x for x in fdb.features(name, 'Center')]
        feature_data_set[name] = {'Left': fl, 'Right': fr, 'Center': fc}

    for name in feature_data_set.keys():
        for pos in feature_data_set[name].keys():
            feature_data_set[name][pos] = anormal.filter(feature_data_set[name][pos]).tolist()

    for name in feature_data_set.keys():
        _len = 0
        for feas in feature_data_set[name].values():
            _len += len(feas)
        if _len == 0:
            del feature_data_set[name]

    if len(feature_data_set):
        estimate_feature_dist()
        train_LOF()


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
