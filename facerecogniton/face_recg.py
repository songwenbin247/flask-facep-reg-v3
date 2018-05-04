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

hot_faces= hot_pool.HotFaces(10)
#FRGraph = FaceRecGraph();
aligner = AlignCustom();
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

def recog_process_frame_offline(fid, frame):
    cameras = []
    rets = []
    aligns = []
    positions = []
    face_locations = []

    rects, landmarks = face_detect.detect_face(frame,40);#min face size is set to 80x80

    if(len(rects)>0):
        aligned_face, face_pos = aligner.align(160,frame,landmarks[0])
        face_locations = np.array(rects[0])
        features_arr = extract_feature.get_features([aligned_face])
        recog_data = findPeople(fid, face_locations, features_arr, [face_pos]);
    else:
        print('mtcnn: no face detected')


    return rets


def recog_process_frame(frames):
    cameras = []
    rets = []
    aligns = []
    positions = []
    face_locations = []

    for (index,frame) in enumerate(frames):
        cameras.append(CameraRouad())
        cameras[index].rects, landmarks = face_detect.detect_face(frame,40);#min face size is set to 80x80
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
        if(len(face_locations)==0):
            face_locations = np.array(c.rects)
        else:
            face_locations = np.concatenate((face_locations, np.array(c.rects)), axis=0)

    if (len(aligns) == 0):
        for (index,camera) in enumerate(cameras):
            rets.append(camera.rets)
        return rets

    features_arr = extract_feature.get_features(aligns)
    recog_data = findPeople(0, face_locations, features_arr, positions);


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
    global LOF_model, KNN_model
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



def findPeople(fid, face_locations, features_arr, positions, thres = 0.6, percent_thres = 97):
    global  mean_intra, var_intra, N_intra, mean_inter, var_inter, N_inter
    global gaus_th, gaus_th0, gaus_rhs
    global total_cnt, false_pos_cnt, false_neg_cnt
    global personal_meta


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


        if (N_intra==0):
            gaus_lhs = 10000
            gaus_rhs = -1000
        else:
            gaus_lhs = np.square(dist-mean_intra)/var_intra - np.square(dist-mean_inter)/var_inter
            #th = (-2)*np.log(gaus_pdf_ratio * np.sqrt(var_intra/var_inter))


        meta = personal_meta[knn_label]


        #svm_pred = meta.svm.predict([features_128D])
        #decision = meta.svm.decision_function([features_128D])

        svm_pred = meta.svm.predict(dist)
        decision = meta.svm.decision_function(dist)


        if (gaus_lhs < gaus_rhs) :
        #if (gaus_lhs < gaus_rhs) and (svm_pred == 1) :

            #face filtering
            #(isHot, isFull) = hot_faces.isHot(knn_label, face_locations[i])
            isHot = 0
            if isHot:
                regRes.append(knn_label)
                #print "person is hot"
                if not isFull:
                    hot_faces.update(knn_label, face_locations[i])
            else:
                total_cnt += 1
                false_neg_cnt += (svm_pred==-1)
                #if (dist < (meta.mean + 0.0*np.sqrt(meta.var))):
                if (svm_pred == 1) or  (dist < (meta.mean - 2.0*np.sqrt(meta.var))):
                    #print('cold pred passd')
                    hot_faces.update(knn_label, face_locations[i])
                    regRes.append(knn_label)
                else:
                    #print('cold pred failed')
                    #print('dist=%f pred=%d, total_cnt=%d, false_neg_cnt=%d, rate=%f' %(dist, svm_pred, total_cnt, false_neg_cnt, (false_neg_cnt+0.0)/total_cnt))
                    #regRes.append("Unknown")
                    regRes.append(" ")

        else:
            #regRes.append("Unknown")
            regRes.append(" ")

        print('fid=%d, ret=[%s], cnt=%d, knn_dist=%1.3f/%1.3f/%d, false_neg=%d, rate=%1.3f, gaus_th=%1.3f/%1.3f, gaus_lhs=%1.3f, gaus_rhs=%1.3f, knn_pred=%s'
            %(fid, regRes, total_cnt, dist, decision, svm_pred, false_neg_cnt, (false_neg_cnt+0.0)/total_cnt,
             gaus_th, gaus_th0,
              gaus_lhs, gaus_rhs, knn_label))

    return regRes


def estimate_feature_dist():
    global  mean_intra, var_intra, N_intra, mean_inter, var_inter, N_inter
    global gaus_rhs, gaus_th, gaus_th0, gaus_rhs, gaus_pdf_ratio
    global personal_metadata, svm_nu, svm_gamma, svm_kernel

    for i in range(len(feature_data_set.keys())):

        # estimate intra-subject distribution
        id = feature_data_set.keys()[i];
        #print('id=%s' %(id))
        personal_meta[id] = FaceMeta()

        vec= np.array([])
        for pos in ['Left','Center', 'Right']:
            personal_data = feature_data_set[id][pos];
            if(len(personal_data) > 0):
                if (len(vec) == 0):
                    vec = np.array(personal_data)
                else:
                    vec = np.concatenate((vec, np.array(personal_data)), axis=0)

        personal_meta[id].centroid = np.mean(vec, axis=0)

        m = 0
        m_new = 0
        var = 0
        n = 0
        var_new =0
        d_sample=[]
        for v in vec:

            d = distance.euclidean(v, personal_meta[id].centroid)
            d_sample.append([d])

            # update mean, variance and count
            mean_intra_new = (N_intra * mean_intra +  d)/(N_intra+1)
            var_intra = (N_intra* var_intra + N_intra * np.square(mean_intra_new - mean_intra)  + np.square(d - mean_intra_new) )/(1+N_intra)
            mean_intra = mean_intra_new
            N_intra = N_intra+1

            #subject-specific mean and variance estimate

            m_new = (n*m + d)/(n+1)
            var = (n*var + n*np.square(m_new - m) + np.square(d - m_new))/(n+1)
            m = m_new
            n = n + 1


        #train a one-class SVM
        #model = OneClassSVM(nu=0.05, kernel="rbf", gamma='auto')
        #model = model.fit(vec)

        model = OneClassSVM(nu=svm_nu, kernel=svm_kernel, gamma=svm_gamma, degree=4)
        model = model.fit(d_sample)

        svm_region = []
        y_prev = -1
        for i in range(1000):
            y_cur = model.predict(i*0.001)
            if ((y_cur==1) and (y_prev == -1)) or ((y_cur==-1) and (y_prev == 1)):
                svm_region.append(i*0.001)
            y_prev = y_cur

        if (len(svm_region)==1):
            svm_region.append(1)

        personal_meta[id].mean = m
        personal_meta[id].var = var
        personal_meta[id].svm = model
        personal_meta[id].svm_region = svm_region




        #print('N_intra=%d, mean=%f, var=%f '%(N_intra, mean_intra, var_intra))

    ## Estimate inter-subject distribution
    for i in range(len(feature_data_set.keys())):
        id1 = feature_data_set.keys()[i];
        v1 = personal_meta[id1].centroid
        for j in range(i+1, len(feature_data_set.keys())):

            id2 = feature_data_set.keys()[j];
            v2 = personal_meta[id2].centroid
            d = distance.euclidean(v1, v2)
            #print('id=%s, d=%1.4f' %(id, d))

            mean_inter_new = (N_inter * mean_inter +  d)/(N_inter+1)

            # update mean, variance and count
            var_inter = (N_inter* var_inter + N_inter * np.square(mean_inter_new - mean_inter)  + np.square(d - mean_inter_new) )/(1+N_inter)
            mean_inter = mean_inter_new
            N_inter = N_inter+1

            #print('(%d,%d), N_inter=%d, d=%f, mean=%f, var=%f '%(i,j, N_inter, d, mean_inter, var_inter))

    gaus_rhs = (-2)*np.log(gaus_pdf_ratio * np.sqrt(var_intra/var_inter))
    a = var_inter - var_intra
    b = -2*(mean_intra * var_inter - mean_inter*var_intra)
    c = mean_intra*mean_intra*var_inter - mean_inter*mean_inter*var_intra - gaus_rhs*var_inter*var_intra
    gaus_th = (-b+ np.sqrt(b*b-4*a*c))/(2*a)

    gaus_rhs0 = (-2)*np.log(1.0 * np.sqrt(var_intra/var_inter))
    c = mean_intra*mean_intra*var_inter - mean_inter*mean_inter*var_intra - gaus_rhs0*var_inter*var_intra
    gaus_th0 = (-b+ np.sqrt(b*b-4*a*c))/(2*a)

    print('a=%f, b=%f, c=%f, gaus_th=%f/%f, gaus_rhs=%f' %(a,b,c,gaus_th, gaus_th0, gaus_rhs))
    print('N_intra=%d, mean=%f, var=%f, sigma=%f'%(N_intra, mean_intra, var_intra, np.sqrt(var_intra)))
    print('N_inter=%d, mean=%f, var=%f, sigma=%f '%(N_inter, mean_inter, var_inter, np.sqrt(var_inter)))



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
    global  mean_intra, var_intra, N_intra, mean_inter, var_inter, N_inter, svm_nu, svm_gamma, svm_kernel

    vec= np.array([])

    # estimate intra-subject distribution
    for pos in ['Left','Center', 'Right']:
        personal_data = personal_features[pos];
        if (len(vec) == 0):
            vec = personal_data
        else:
            vec = np.concatenate((vec, personal_data), axis=0)

    #Calc cluster centroid
    centroid = np.mean(vec, axis=0)

    #Calc inter and intra subject distribution
    m = 0
    m_new = 0
    var = 0
    n = 0
    var_new =0

    d_sample=[]
    for v in vec:

        d = distance.euclidean(v, centroid)
        d_sample.append([d])

        # update mean, variance and count
        mean_intra_new = (N_intra * mean_intra +  d)/(N_intra+1)
        var_intra = (N_intra* var_intra + N_intra * np.square(mean_intra_new - mean_intra)  + np.square(d - mean_intra_new) )/(1+N_intra)
        mean_intra = mean_intra_new
        N_intra = N_intra+1

        #subject-specific mean and variance estimate
        m_new = (n*m + d)/(n+1)
        var = (n*var + n*np.square(m_new - m) + np.square(d - m_new))/(n+1)
        m = m_new
        n = n + 1

    #train a one-class SVM
    model = OneClassSVM(nu=svm_nu, kernel=svm_kernel, gamma=svm_gamma)
    model = model.fit(d_sample)

    #add it to personal metadata
    id = name
    personal_meta[id] = FaceMeta()
    personal_meta[id].mean = m
    personal_meta[id].var = var
    personal_meta[id].svm = model
    personal_meta[id].centroid = centroid



    print('[New] id=%s, m=%f, var=%f, sigma=%f' %(id, m, var, np.sqrt(var)))

    #print('N_intra=%d, mean=%f, var=%f '%(N_intra, mean_intra, var_intra))

    v1 = personal_meta[id].centroid
    for v2 in vec_global:
        d = distance.euclidean(v1, v2)

        mean_inter_new = (N_inter * mean_inter +  d)/(N_inter+1)

        # update mean, variance and count
        var_inter = (N_inter* var_inter + N_inter * np.square(mean_inter_new - mean_inter)  + np.square(d - mean_inter_new) )/(1+N_inter)
        mean_inter = mean_inter_new
        N_inter = N_inter+1

            #print('N_inter=%d, mean=%f, var=%f '%(N_inter, mean_inter, var_inter))

    gaus_rhs = (-2)*np.log(gaus_pdf_ratio * np.sqrt(var_intra/var_inter))
    a = var_inter - var_intra
    b = -2*(mean_intra * var_inter - mean_inter*var_intra)
    c = mean_intra*mean_intra*var_inter - mean_inter*mean_inter*var_intra - gaus_rhs*var_inter*var_intra
    gaus_th = (-b+ np.sqrt(b*b-4*a*c))/(2*a)

    gaus_rhs0 = (-2)*np.log(1.0 * np.sqrt(var_intra/var_inter))
    c = mean_intra*mean_intra*var_inter - mean_inter*mean_inter*var_intra - gaus_rhs0*var_inter*var_intra
    gaus_th0 = (-b+ np.sqrt(b*b-4*a*c))/(2*a)

    print('[new] N_intra=%d, mean=%f, var=%f, sigma=%f'%(N_intra, mean_intra, var_intra, np.sqrt(var_intra)))
    print('[new[ N_inter=%d, mean=%f, var=%f, sigma=%f'%(N_inter, mean_inter, var_inter, np.sqrt(var_inter)))

    print('[new] a=%f, b=%f, c=%f, gaus_th=%f/%f, gaus_rhs=%f' %(a,b,c,gaus_th, gaus_th0, gaus_rhs))


def save_feature_Local(name, person_features):

    #update estimate using existing dataset
    update_feature_dist(name, person_features);

    feature_data_set[name] = person_features;

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
        person_features[pos] = extract_feature.get_features(person_imgs[pos]).tolist()

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

def train_process_people(frames):
    frame = frames[0]
    rects, landmarks = face_detect.detect_face(frame, 20);
    ret_per_frame = []
    rets = []
    if (len(rects) == 1):
        aligned_frame, face_pos = aligner.align(160,frame,landmarks[0]);
        if (len(person_imgs[face_pos]) < 20):
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

    if len(feature_data_set):
        estimate_feature_dist()
        train_LOF()

    for id in personal_meta.keys():
        svm_region = personal_meta[id].svm_region
        print('id=%s, m=%1.3f, var=%1.3f, sigma=%1.3f, svm_region=%s, sigma*[%1.3f, %1.3f]'
                %(id, personal_meta[id].mean, personal_meta[id].var, np.sqrt( personal_meta[id].var ),
                    svm_region,
                    (svm_region[0] - personal_meta[id].mean)/np.sqrt( personal_meta[id].var),
                    (svm_region[1] - personal_meta[id].mean)/np.sqrt( personal_meta[id].var)
                 ))


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
