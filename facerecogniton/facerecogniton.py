import json
from multiprocessing import Process, Queue, Lock, Manager
import threading, time
import numpy as np
import paho.mqtt.client as mqtt

Global = Manager().Namespace()

CMD_TRAIN_START   = 0
CMD_TRAIN_FINISH  = 1
CMD_TRAIN_STATUS = 2
CMD_DELETE_NAME = 3
CMD_GET_NAMES = 4

Global.poscount = {"Left" : 0, "Right": 0, "Center": 0};
class FaceRecognitonProcess(Process):
    def __init__(self, frameq, retq, cmdq, cmdretq):
        Process.__init__(self)
        self.frameq = frameq
        self.retq = retq
        self.cmdq = cmdq
        self.cmdretq = cmdretq
        self.training = 0
        self.soundmqtt = mqtt.Client()
        self.history_names = {}

    def sendGuidence(self, msg):
        self.soundmqtt.publish("NXP_CMD_SOUND_GUIDE", msg)

    def sendResult(self, ret):
        self.retq.put_nowait(ret)

    def reciveFrame(self):
        return self.frameq.get_nowait()

    def training_callback(self):
        self.training = 0

    def run(self):
        import face_recg as face_recg
        self.soundmqtt.connect("localhost", 1883, 60)
        self.soundmqtt.loop_start()
        
        print("Face recognition engine initialized")
        print("Please open browser and visite https://[board-ip]:5000/")
        while (1):
            try:
                if self.cmdq.full():
                    cmd, param = self.cmdq.get()
                    if cmd == CMD_TRAIN_START:
                        print("CMD_TRAIN_START")
                        if self.training == 0:
                            rets = face_recg.train_start(param)
                            self.training = 1
                            print("Send turn face")
                            self.sendGuidence("turn_face")
                        else:
                            rets = False
                    elif cmd == CMD_TRAIN_FINISH:
                        print("CMD_TRAIN_FINISH")
                        if self.training == 1:
                            rets = face_recg.train_finish(self.training_callback)
                            self.training = 2
                            self.sendGuidence("training")
                        else:
                            rets = False
                    elif cmd == CMD_TRAIN_STATUS:
                        if self.training == 0:
                            rets = False
                            self.sendGuidence("train_over")
                        else:
                            rets = True
                    elif cmd == CMD_DELETE_NAME:
                        print("CMD_DELETE_NAME")
                        rets = face_recg.delete_name(param)
                    elif cmd == CMD_GET_NAMES:
                        print("CMD_GET_NAMES")
                        rets = face_recg.get_names()
                    self.cmdretq.put(rets)
                elif self.frameq.full():
                    inFrame= self.reciveFrame()
                    if self.training == 1:
                        rets = face_recg.train_process_people(inFrame)
                    elif self.training == 0:
                        rets = face_recg.recog_process_frame(inFrame)
                        if len(rets) == 1 and rets[0]["pos"] == "Center" and rets[0]["name"] != " ":
                            name = rets[0]["name"]
                            if name not in self.history_names:
                                self.history_names[name] = [0, int(time.time() * 1000)]
                            else:
                                self.history_names[name][0] += 1

                            if self.history_names[name][0] == 5:
                                self.sendGuidence(name)
                    else:
                        rets = face_recg.detect_people(inFrame)
                    self.sendResult(rets)
            except Exception as e:
                pass

frameq = None
retq = None
cmdq = None
cmdretq = None



def initEngine():
    global frameq,retq, cmdq,cmdretq
    
    frameq = Queue(maxsize = 1)
    retq = Queue(maxsize = 1)
    cmdq = Queue(maxsize = 1)
    cmdretq = Queue(maxsize = 1)
    process = FaceRecognitonProcess(frameq, retq, cmdq, cmdretq)
    process.start()

def proImageFile(imgf):
    frame = np.array(imgf)
    proCvFrame(frame)

def proCvFrame(frame):
    try:
        if (frameq.full()):
            frameq.get_nowait()
        frameq.put(frame)
    except Exception as e:
        print(e)

def trainStart(name):
    cmdq.put((CMD_TRAIN_START, name))
    ret = cmdretq.get()
    return ret

def getTrainStatus():
    cmdq.put((CMD_TRAIN_STATUS, None))
    ret = cmdretq.get()
    return ret

def trainFinish():
    cmdq.put((CMD_TRAIN_FINISH, None))
    ret = cmdretq.get()
    return ret

def getResult():
    try:
        rets= retq.get_nowait()
        return rets
    except Exception as e:
        return None

def deleteName(name):
    cmdq.put((CMD_DELETE_NAME, name))
    ret = cmdretq.get()
    return ret

def getNames():
    cmdq.put((CMD_GET_NAMES, None))
    ret = cmdretq.get()
    return ret

initEngine()
