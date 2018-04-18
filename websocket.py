import json
from twisted.internet import reactor
from autobahn.twisted.websocket import WebSocketServerFactory, \
                               WebSocketServerProtocol, \
                               listenWS
from twisted.internet import task, defer

from twisted.internet.ssl import DefaultOpenSSLContextFactory
import facerecogniton.facerecogniton as facerecg
from multiprocessing import Process
import time, StringIO, base64, os
from PIL import Image
import threading
import paho.mqtt.client as mqtt


VIDEO_DEVICE = ""
WEBSOCKET_PORT = 9000
 
class FaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(FaceServerProtocol, self).__init__()
        self.new_person = None

    def modulesUpdate(self, payload):
        names = facerecg.moduleUpdate(payload)
        self.sendSocketMessage("LOADNAME_RESP", ",".join(names))

    def onOpen(self):
        self.factory.register(self)

    def onClose(self, wasClean, code, reason):
        pass

    def getTrainStatus(self):
        ret = facerecg.getTrainStatus()
        if ret[0] == 1:
            #recoding
            reactor.callLater(0.5, self.getTrainStatus)
            self.sendSocketMessage("TRAINPROCESS", ret[1])
        elif ret[0] == 2:
            #training on local processing
            reactor.callLater(0.5, self.getTrainStatus)
        elif ret[0] == 0:
            #training on local finished
            self.sendSocketMessage("TRAINFINISH_RESP")

    def onMessage(self, payload, binary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        if msg['type'] == "CONNECT_REQ":
            self.sendSocketMessage("CONNECT_RESP", msg)
        elif msg['type'] == "LOADNAME_REQ":
            names = facerecg.getNames()
            self.sendSocketMessage("LOADNAME_RESP", ",".join(names))
            if VIDEO_DEVICE == "laptop":
                self.sendSocketMessage("INITCAMERA")
            else:
                self.sendSocketMessage("INITVIDEO")
        elif msg['type'] == "DELETENAME_REQ":
            name = msg['msg']
            ret = facerecg.deleteName(name)
            if (ret != True):
                self.sendSocketMessage("ERROR_MSG", name + " is not in database")
        elif msg['type'] == "RECGFRAME_REQ":
            self.proWebFrame(msg['dataURL'])
            ret = facerecg.getResult()
            if ret is not None:
                self.sendSocketMessage("RECGFRAME_RESP", ret[0])
        elif msg['type'] == "TRAINSTART_REQ":
            name = msg['msg']
            ret = facerecg.trainStart(name)
            if (ret != True):
                self.sendSocketMessage("ERROR_MSG", name + " is already in database or training is not finished")
            else:
                self.sendSocketMessage("TRAINSTART_RESP")
                reactor.callLater(0.5, self.getTrainStatus)
        elif msg['type'] == "RECODFINISH_REQ":
            facerecg.recodFinish()

    def sendSocketMessage(self, mtype, msg = ""):
        msg = { "type" : mtype, 'msg' : msg }
        self.sendMessage(json.dumps(msg))

    def proWebFrame(self, dataURL):
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgf = Image.open(StringIO.StringIO(imgdata))
        facerecg.proImageFile(imgf)

class FaceModuleFactory(WebSocketServerFactory):
    def __init__(self, *args, **kwargs):
        super(FaceModuleFactory, self).__init__(*args, **kwargs)
        self.client = None

    def register(self, client):
        self.client = client

    def modulesUpdate(self, client, data, msg):
        self.client.modulesUpdate(msg.payload)

fdir = os.path.dirname(os.path.realpath(__file__))
tls_crt = os.path.join(fdir, 'tls', 'server.crt')
tls_key = os.path.join(fdir, 'tls', 'server.key')


def startSocketServer(serverip):
    facerecg.initEngine(serverip)
    factory = FaceModuleFactory()

    if serverip != None:
        broadmqtt = mqtt.Client()
        broadmqtt.connect(serverip, 1883, 60)
        broadmqtt.subscribe("NXP_CMD_MODULE_UPDATE", qos=1)
        broadmqtt.on_message = factory.modulesUpdate
        broadmqtt.loop_start()

    #factory = WebSocketServerFactory()
    factory.protocol = FaceServerProtocol
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(WEBSOCKET_PORT, factory, ctx_factory)
    reactor.run()

def startWebSocketServer(serverip):
    p2 = Process(target = startSocketServer, args=(serverip, ))
    p2.start()

