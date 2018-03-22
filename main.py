#!/usr/bin/env python
from importlib import import_module
import os,time
from flask import Flask, render_template, Response
import argparse
import websocket

parser = argparse.ArgumentParser()
parser.add_argument('--dev', type=str, required=True,
                    help='[usb|laptop|"url" of IP camera]input video device')
parser.add_argument('--httpport', type=int,
                    help='The port for http server')
parser.add_argument('--socketport', type=int,
                    help='The port for websocket server')
args = parser.parse_args()

print("Initialzing face recognition engine.")
if args.dev == 'laptop':
    print("Using the camera of laptop.")
elif args.dev == 'usb':
    from camera_opencv import *
    print("Using onboard usb camera")
else:
    from camera_opencv import *
    Camera.set_video_source(args.dev)
    websocket.VIDEO_DEVICE = args.dev
    print("Using ip camera with url(%s)" % args.dev)

if args.httpport != None:
    HTTP_PORT = args.httpport
else:
    HTTP_PORT = 5000

if args.socketport != None:
    websocket.WEBSOCKET_PORT = args.socketport
else:
    websocket.WEBSOCKET_PORT = 9000
    
import re
def alter(old_file, new_file, old_str, new_str):
    with open(old_file, "r") as f1,open(new_file, "w") as f2:
        for line in f1:
            f2.write(re.sub(old_str,new_str,line))
    f1.close()
    f2.close()

alter("templates/index_static.html", "templates/index_web.html", "WEBSOCKET_PORT", str(websocket.WEBSOCKET_PORT))   

fdir = os.path.dirname(os.path.realpath(__file__))
tls_crt = os.path.join(fdir, 'tls', 'server.crt')
tls_key = os.path.join(fdir, 'tls', 'server.key')

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    #return render_template('index_opencv.html')
    return render_template('index_web.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

@app.route('/videoel')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if args.dev == 'laptop':
        return Response()
    else:
        print("video_feed")
        return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    websocket.startWebSocketServer(tls_key, tls_crt)
    #app.run(host='0.0.0.0', threaded=True)
    app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True, ssl_context=(tls_crt, tls_key))

