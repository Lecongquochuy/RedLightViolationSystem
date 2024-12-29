from flask import Flask, render_template, Response, jsonify, request
import os
import cv2

app = Flask(__name__)

app.config['SECRET_KEY'] = 'lecongquochuy'
app.config['UPLOAD_FOLDER'] = '../yolov10/data_video'

def generate_frame(path):
    yolov10_output = objectTracking(path)
    for im0, frameRate, frameShape, totalDetectionm in yolov10_output:
        ret, butter = cv2.imencode('.jpg', im0)
        frame = butter.tobytes()
