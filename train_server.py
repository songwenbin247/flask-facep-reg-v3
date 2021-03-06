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
    def __send_train_finish(self, name):
        print "__send_train_finish"
        mqttclient.publish("NXP_CMD_MODULE_UPDATE", name)

    def __send_delete_name(self, name):
        print "__send_delete_name"
        mqttclient.publish("NXP_CMD_MODULE_UPDATE", name)

    def get(self):
        return send_from_directory(modulespath, 'facerec_128D.txt', as_attachment=True)

    def delete(self):
        args = parser.parse_args()
        name = args['id'].decode('utf-8')
        ret = facemodules.delete_module(name)
        if ret == True:
            self.__send_delete_name(name)
            return {'state':'SUCCESS'}, 200
        elif ret == False:
            return {'state':'FAILED'}, 200

    def put(self):
        return {'state':'SUCCESS'}, 201

    def post(self):
        args = parser.parse_args()
        name = args['id'].decode('utf-8')
        ret = facemodules.training_start(name)

        if ret == False:
            print "already in"
            return {'state':'EXIST'}, 201

        for fname, f in request.files.items():
            pif=StringIO.StringIO()
            f.save(pif)
            pif.seek(0)
            pili = Image.open(pif)
            frame = np.array(pili)
            print fname
            if "Center" in fname:
                facemodules.training_proframe(name, frame, "Center")
            elif "Right" in fname:
                facemodules.training_proframe(name, frame, "Right")
            elif "Left" in fname:
                facemodules.training_proframe(name, frame, "Left")
            else:
                print("Warn:invalid position")
        ret = facemodules.training_finish(name, self.__send_train_finish)

        if ret == True:
            print "return success"
            return {'state':'SUCCESS'}, 201
        elif ret == False:
            print "return false"
            return {'state':'FAILED'}, 201

##
## Actually setup the Api resource routing here
##
api.add_resource(TrainModels, '/train')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8383, debug=False)
