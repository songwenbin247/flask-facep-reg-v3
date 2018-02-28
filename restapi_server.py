#encoding=utf-8
#!/usr/bin/env python
# by vellhe 2017/7/9
from flask import Flask, send_from_directory
from flask_restful import reqparse, abort, Api, Resource, request
import werkzeug
import os
import StringIO
import paho.mqtt.client as mqtt
import numpy as np
from PIL import Image



import facerecogniton.facemodules_server as facemodules
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
parser = reqparse.RequestParser()

app = Flask(__name__)
api = Api(app)
parser.add_argument('id', type=str, location='args')
parser.add_argument('end', type=str, location='args')
parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')

mqttclient = mqtt.Client()
mqttclient.connect('localhost', 1883, 60)
mqttclient.loop_start()

modulespath = os.path.join(app.root_path,'models')
class TrainModels(Resource):
    def __modules_updated(self):
        mqttclient.publish("NXP_FACE_RECG_MODULES_UPDATED", "True")

    def get(self):
        return send_from_directory(modulespath, 'facerec_128D.txt', as_attachment=True)

    def delete(self):
        args = parser.parse_args()
        ret = facemodules.delete_module(args['id'].decode('utf-8'))
        if ret == True:
            ret = {'state':'SUCCESS'}
            self.__modules_updated()
        elif ret == False:
            ret = {'state':'FAILED'}
        return ret, 200

    def put(self):
        args = parser.parse_args()
        ret = facemodules.training_start(args['id'].decode('utf-8'))
        if ret == True:
            ret = {'state':'SUCCESS'}
        elif ret == False:
            ret = {'state':'FAILED'}
        return ret, 201

    def post(self):
        args = parser.parse_args()
        for fname, f in request.files.items():
            pif=StringIO.StringIO()
            f.save(pif)
            pili = Image.open(pif)
            frame = np.array(pili)
            facemodules.training_proframe(args['id'].decode('utf-8'), frame)
        ret = facemodules.training_finish(args['id'].decode('utf-8'), self.__modules_updated)
        if ret == True:
            ret = {'state':'SUCCESS'}
        elif ret == False:
            ret = {'state':'FAILED'}
        return ret, 201

##
## Actually setup the Api resource routing here
##
api.add_resource(TrainModels, '/train')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8383, debug=False)
