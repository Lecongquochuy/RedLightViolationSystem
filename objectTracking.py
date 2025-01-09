#Import All the Required Libraries
import cv2
import math
import time
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLOv10
from utils.object_tracking import ObjectTracking
from utils.NonMaximumSuppression import non_max_suppression_fast
from utils.nms import apply_nms
from utils.preprocess_img import preprocess_image
from utils.calculate_direction import calculate_direction
from utils.sort import Sort

class objectTracking:
    def __init__(self, model_path, plate_model_path):
        self.model = YOLOv10(model_path)
        self.plate_model = YOLOv10(plate_model_path)
        self.tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)
        self.object_Tracking = ObjectTracking()
        self.deepsort = self.object_Tracking.initialize_deepsort()
        self.classNames = ["car", "green_light", "motobike", "red_light", "stop_line", "yellow_light", "License_Plate"]
        self.limits = [0, 0, 0, 0]
        self.left_x1 = 1e9
        self.left_y1 = 1e9
        self.right_x2 = 0
        self.right_y2 = 0
        self.notErrorIDs = set()
        self.crossedIDs = set()
        self.totalError = 0

    def detect_license_plate(self, frame):
        if not self.plate_model:
            return frame
        results = self.plate_model.predict(frame, conf=0.3, iou=0.5)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            label = f"License Plate: {conf}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def process_frame(self, frame, count):
        # cropped_frame = frame[0:new_height, pixels_to_cut:frame_width]
        # resized_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
        frame = preprocess_image(frame)
        check_red_light = 0
        detections = np.empty((0, 5))
        xywh_bboxs = []
        confs = []
        oids = []

        results = self.model.predict(frame, conf=0.25, iou=0.7)
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            bbox_width = abs(x1 - x2)
            bbox_height = abs(y1 - y2)
            xcycwh = [cx, cy, bbox_width, bbox_height]

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = self.classNames[cls]
            label = f"{currentClass} : {conf}"
            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
            c2 = x1 + textSize[0] + 24, y1 - textSize[1]

            if currentClass == "stop_line" and conf > 0.3:
                self.left_x1 = min(self.left_x1, x1)
                self.left_y1 = min(self.left_y1, y1)
                self.right_x2 = max(self.right_x2, x2)
                self.right_y2 = max(self.right_y2, y2)

                # distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # if distance > max_distance:
                #     max_distance = distance
                self.limits = [self.left_x1, int((self.left_y1 + self.right_y2) / 2), self.right_x2, int((self.left_y1 + self.right_y2) / 2)]

            if currentClass not in ["green_light", "red_light", "stop_line", "yellow_light"] and conf > 0.25:
                xywh_bboxs.append(xcycwh)
                confs.append(conf)
                oids.append(cls)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

            if currentClass == "red_light" and conf > 0.2 and check_red_light != 2:
                check_red_light = 1

            if currentClass in ["yellow_light", "green_light"] and conf > 0.2:
                check_red_light = 2

            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 16, y1), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        detections = non_max_suppression_fast(detections, iou_threshold=0.5)
        resultsTracker = self.tracker.update(detections)
        cv2.line(frame, (self.limits[0], self.limits[1] - 4), (self.limits[2], self.limits[3] + 8), (0, 255, 0), 5)
        return frame, resultsTracker, check_red_light
    def track_object(self, frame, resultsTracker, check_red_light):
        # if check_red_light == 1:
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2 + 6), (255, 0, 0), 1)
            cv2.putText(frame, f'{int(id)}', (x1, y1 - 2), 0, 0.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            if id not in self.notErrorIDs and check_red_light == 1:
                if y2 > self.limits[1]:
                    self.notErrorIDs.add(id)

            if id not in self.crossedIDs and check_red_light == 1:
                if y2 < self.limits[1] and (id in self.notErrorIDs):
                    self.crossedIDs.add(id)
                    self.notErrorIDs.remove(id)
                    cv2.line(frame, (self.limits[0], self.limits[1]), (self.limits[2], self.limits[3]), (0, 0, 255), 5)
        return frame