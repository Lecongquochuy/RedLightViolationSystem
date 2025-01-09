import cv2
import math
import time
import os
from ultralytics import YOLOv10
from utils.preprocess_img import preprocess_image

video_path = r"D:\KhoaLuan\yolov10\data_video\video_track_1.mp4"
model_path = r"D:\KhoaLuan\yolov10\yolov10\weights\best_addLight.pt"

cap = cv2.VideoCapture(video_path)
model = YOLOv10(model_path)
classnames = ["car", "green_light", "motobike", "red_light", "stop_line", "yellow_light"]
color = []
ctime = 0
ptime = 0
count = 0
pixels_to_cut = 300
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = frame_width - 2 * pixels_to_cut
new_height = frame_height - 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cropped_frame = frame[0:new_height, pixels_to_cut:frame_width]
    resized_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

    # frame = cropped_frame
    frame = preprocess_image(resized_frame)
    count += 1
    print(f"Frame: {count}")
    results = model.predict(source=frame, conf=0.3, save=False)
    # for result in results:
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        conf = math.ceil(box.conf[0] * 100) / 100
        cls = int(box.cls[0])
        classname = classnames[cls]
        label = f"{classname} : {conf}"
        textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        c2 = x1 + textSize[0], y1 - textSize[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(frame, f"Frame: {str(count)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()

    # for result in results:
    #     boxes = result.boxes
    #     for box in boxes:
    #         x1, y1, x2, y2 = box.xyxy[0]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #         conf = math.ceil(box.conf[0] * 100) / 100
    #         cls = int(box.cls[0])
    #         classname = classnames[cls]
    #         label = f"{classname} : {conf}"
    #         textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
    #         c2 = x1 + textSize[0], y1 - textSize[1] - 3
    #         cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
    #         cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    #     ctime = time.time()
    #     fps = 1 / (ctime - ptime)
    #     ptime = ctime
    #     cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    #     cv2.putText(frame, f"Frame Count: {str(count)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    #     cv2.imshow("Frame", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('1'):
    #         break
    #     else:
    #         break
# cap.release()
# cv2.destroyAllWindows()