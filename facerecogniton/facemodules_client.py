'''
Main program
@Author: David Vu

To execute simply run:
main.py

To input new user:
main.py --mode "input"

'''

import cv2
import argparse
import sys, time
import json
import numpy as np
from PIL import Image
import StringIO
import requests
import threading

url = ''
person_images = {}
feature_data_set = {}
info_data_set = {}

def get_names():
    names = []
    for name in feature_data_set:
        names.append(name)
    return names

def delete_module(name):
    if (name not in feature_data_set):
        return False
    args = {'id': name}
    headers = {"Content-type":"application/json","Accept": "application/json"}
    r = requests.delete(url, params=args, headers=headers)
    ret = json.loads(r.text)
    if ('state'in ret and ret['state'] == 'SUCCESS'):
        return True

def __training_thread(name, callback):
    print "__training_thread"
    args = {'id': name, 'end':'true'}
    headers = {"Content-type":"application/json","Accept": "application/json"}
    files = {}
    for i,f in enumerate(person_images[name]):
        files['file{}'.format(i)] = ('{}.jpeg'.format(i), f, 'image/jpeg')
    r = requests.post(url, params=args, files=files)
    del person_images[name]
    if callback:
        callback()

def training_start(name):
    if (name in feature_data_set):
        return False
    args = {'id': name}
    person_images[name] = []
    headers = {"Content-type":"application/json","Accept": "application/json"}
    r = requests.put(url, params=args, headers=headers)
    ret = json.loads(r.text)
    if ('state'in ret and ret['state'] == 'SUCCESS'):
        return True
    else:
        return False

def training_proframe(name, frame):
    if name not in person_images:
        person_images[name] = []
    picf = StringIO.StringIO()
    pi = Image.fromarray(frame)
    pi.save(picf, format = "jpeg")
    picf.seek(0)

    person_images[name].append(picf)

def training_proimage(name, img):
    person_images[name].append(img)

def training_finish(name, callback=None):
    print("in training_finish", name)
    t = threading.Thread(target=__training_thread, args=(name, callback,))
    t.start()
    return t

def update_modules():
    print "Downloading modules from cloud."
    global feature_data_set, info_data_set
    headers = {"Content-type":"application/json","Accept": "application/json"}
    r = requests.get(url, headers=headers)
    if (r.status_code == 200):
        f = open('./models/facerec_128D.txt','w')
        f.write(r.content)
        f.close()
        feature_data_set = json.loads(r.content)

        f = open('./models/detailinfo.txt','r')
        text = f.read()
        info_data_set = json.loads(text)
        f.close()
        print "Modules updated successfully"
        return True
    else:
        print "Modules updated failed"
        return False

def has_name(name):
    return name in feature_data_set

def get_info(name):
    if name in info_data_set:
        return info_data_set[name]
    else:
        return ""

def modules_init(serverip='localhost'):
    global url
    url  = 'http://{}:8383/train'.format(serverip)
    update_modules()
