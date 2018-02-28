#!/usr/bin/env python
from importlib import import_module
import os,time
from flask import Flask, render_template, Response
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', type=str, required=True,
                    help='[usb|laptop|"url" of IP camera]input video device')
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
    print("Using ip camera with url(%s)" % args.dev)
    
from websocket import startWebSocketServer

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
    startWebSocketServer(tls_key, tls_crt, args.dev)
    #app.run(host='0.0.0.0', threaded=True)
    app.run(host='0.0.0.0', threaded=True, ssl_context=(tls_crt, tls_key))

