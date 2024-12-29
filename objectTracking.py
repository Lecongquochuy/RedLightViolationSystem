#Import All the Required Libraries
import cv2
import math
import time
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLOv10
from utils.object_tracking import ObjectTracking
from yolov10.utils.NonMaximumSuppression import non_max_suppression_fast
from yolov10.utils.nms import apply_nms
from yolov10.utils.preprocess_img import preprocess_image
from yolov10.utils.calculate_direction import calculate_direction
from yolov10.utils.sort import Sort

objectTracking = ObjectTracking()
deepsort = objectTracking.initialize_deepsort()
tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)

video_path = r"D:\KhoaLuan\yolov10\data_video\video4.mp4"
model_path = r"D:\KhoaLuan\yolov10\yolov10\weights\best_addLight.pt"
output_csv_path = r"D:\KhoaLuan\yolov10\tracking_output_5.csv"

#Create a Video Capture Object
cap = cv2.VideoCapture(video_path)
model = YOLOv10(model_path)
classNames = ["car", "green_light", "motobike", "red_light", "stop_line", "yellow_light"]
# classNames = ['bicycle', 'bus', 'car', 'green_light', 'motobike', 'red_light', 'stop_line', 'truck', 'yellow_light']

frame_interval = 3
threshold_distance = 5
ctime = 0
ptime = 0
count = 0
limits = [0, 0, 0, 0]
left_x1 = 1e9
left_y1 = 1e9
right_x2 = 0
right_y2 = 0
max_conf = 0
max_distance = 0
tracking_data = []
last_positions = {}
trajectories = {}
crossed_line = {}  # Lưu tọa độ khi xe chạm vạch
lost_ids = set()
pixels_to_cut = 300
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = frame_width - 2 * pixels_to_cut
new_height = frame_height - 200

totalCount = []
crossedIDs = set()
notErrorIDs = set()
totalError = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cropped_frame = frame[0:new_height, pixels_to_cut:frame_width]
    resized_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
    frame = preprocess_image(resized_frame)
    count += 1
    print(f"Frame Count: {count}")

    # results = model.predict(frame, conf=0.25)
    results = model.predict(frame, conf=0.25, iou=0.7)
    result = results[0]
    boxes = result.boxes

    check_red_light = 0
    detections = np.empty((0, 5))
    xywh_bboxs = []
    confs = []
    oids = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        bbox_width = abs(x1 - x2)
        bbox_height = abs(y1 - y2)
        xcycwh = [cx, cy, bbox_width, bbox_height]

        conf = math.ceil(box.conf[0] * 100) / 100
        cls = int(box.cls[0])
        currentClass = classNames[cls]
        label = f"{currentClass} : {conf}"
        textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        c2 = x1 + textSize[0] + 24, y1 - textSize[1]

        if currentClass == "stop_line" and conf > 0.3:
            left_x1 = min(left_x1, x1)
            left_y1 = min(left_y1, y1)
            right_x2 = max(right_x2, x2)
            right_y2 = max(right_y2, y2)

            # distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # if distance > max_distance:
            #     max_distance = distance
            limits = [left_x1, int((left_y1 + right_y2) / 2), right_x2, int((left_y1 + right_y2) / 2)]

        if currentClass != "green_light" and currentClass != "red_light" and currentClass != "stop_line" \
                and currentClass != "yellow_light" and conf > 0.25:
            xywh_bboxs.append(xcycwh)
            confs.append(conf)
            oids.append(cls)
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

        if currentClass == "red_light" and conf > 0.2 and check_red_light != 2:
            check_red_light = 1

        if currentClass == "yellow_light" or currentClass == "green_light" and conf > 0.2:
            check_red_light = 2

        cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 16, y1), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    detections = non_max_suppression_fast(detections, iou_threshold=0.5)
    resultsTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1] - 4), (limits[2], limits[3] + 8), (0, 255, 0), 5)

    if check_red_light == 1:
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1

            if id not in notErrorIDs:
                if y2 > limits[1]:
                    notErrorIDs.add(id)

            if id not in crossedIDs:
                if y2 < limits[1] and (id in notErrorIDs):
                    crossedIDs.add(id)
                    notErrorIDs.remove(id)
                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(frame, (x1, y1), (x2, y2 + 6), (255, 0, 0), 1)
        cv2.putText(frame, f'{int(id)}', (x1, y1 - 2), 0, 0.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        if id in crossedIDs:
            totalError = len(crossedIDs)
            cv2.putText(frame, f'Violated', (max(0, x1) - 8, max(35, y1) - 16), 0, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.putText(frame, f'Total : {totalError}', (25, 25), 0, 1, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

# Save tracking data to CSV
df = pd.DataFrame(tracking_data)
df.to_csv(output_csv_path, index=False)
cap.release()
cv2.destroyAllWindows()

    # label = f"{currentClass} : {conf}"
    # cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
    # cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # # xywh_bboxs, confs = apply_nms(xywh_bboxs, confs, iou_threshold=0.9)
    # xywhs = torch.tensor(xywh_bboxs).float()
    # confidence = torch.tensor(confs).float()
    # outputs = deepsort.update(xywhs, confidence, oids, frame)
    #
    # current_ids = set()
    # if len(outputs) > 0:
    #     bbox_xyxy = outputs[:,:4]
    #     identities = outputs[:,-2]
    #     classID = outputs[:,-1]
    #     objectTracking.draw_boxes(frame, bbox_xyxy, identities, classID)
    #
    #     for output in outputs:
    #         x1, y1, x2, y2, obj_id, cls = output
    #         cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    #
    #         if obj_id not in trajectories:
    #             trajectories[obj_id] = []
    #         trajectories[obj_id].append((cx, cy))
    #
    #         # Nếu đối tượng chạm vạch
    #         if obj_id not in crossed_line and limits[0] < cx < limits[2] and abs(cy - limits[1]) < 5:
    #             crossed_line[obj_id] = (cx, cy)  # Lưu tọa độ khi chạm vạch
    #             tracking_data.append({
    #                 "Frame": count,
    #                 "ObjectID": obj_id,
    #                 "Class": classNames[int(cls)],
    #                 "X": cx,
    #                 "Y": cy,
    #                 "Width": int(x2 - x1),
    #                 "Height": int(y2 - y1),
    #                 "Status": "Crossed Line"
    #             })
    #
    #         current_ids.add(obj_id)
    #
    # # Tìm đối tượng bị mất ID
    # lost_ids = set(trajectories.keys()) - current_ids
    # for lost_id in lost_ids:
    #     if lost_id in crossed_line:  # Đối tượng đã chạm vạch
    #         last_position = trajectories[lost_id][-1]  # Tọa độ cuối trước khi mất
    #         start_position = crossed_line[lost_id]  # Tọa độ lúc chạm vạch
    #         direction = calculate_direction([start_position, last_position])  # Tính hướng di chuyển
    #
    #         tracking_data.append({
    #             "Frame": count,
    #             "ObjectID": lost_id,
    #             "Class": "N/A",
    #             "X": last_position[0],
    #             "Y": last_position[1],
    #             "Width": 0,
    #             "Height": 0,
    #             "Direction": direction,
    #             "Status": "Lost ID"
    #         })
    # cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('1'):
#         break
#
# # Save tracking data to CSV
# df = pd.DataFrame(tracking_data)
# df.to_csv(output_csv_path, index=False)
# cap.release()
# cv2.destroyAllWindows()
