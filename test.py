import cv2
import numpy as np
import math
import time
import os
from ultralytics import YOLOv10

video_path = r"D:\KhoaLuan\yolov10\data_video\Recording-2024-08-19-214233.mp4"
model_path = r"D:\KhoaLuan\yolov10\yolov10\weights\best.pt"

cap = cv2.VideoCapture(video_path)
model = YOLOv10(model_path)
classnames = ["car", "green_light", "motobike", "red_light", "stop_line", "yellow_light"]
color = []
ctime = 0
ptime = 0
count = 0
def preprocess_image(image):
    # Tăng cường độ sáng
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 10)  # Tăng độ sáng
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
def increase_brightness(image, value=20):
    # Dùng numpy để cộng giá trị cho tất cả các pixel
    image_bright = cv2.convertScaleAbs(image, alpha=1, beta=value)
    return image_bright

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    print(f"Frame: {count}")

    # # Chuyển đổi ảnh sang không gian màu RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    # # Tăng độ sáng ảnh
    # bright_image = increase_brightness(frame, value=50)
    #
    # frame = cv2.cvtColor(bright_image, cv2.COLOR_RGB2BGR)
    # frame = bright_image
    frame = preprocess_image(frame)


    results = model.predict(source=frame, conf=0.2, save=False)
    for result in results:
        boxes = result.boxes
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
#
# # Đọc ảnh
# image = cv2.imread('image.jpg')
#
# # Chuyển đổi ảnh sang không gian màu YUV
# yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#
# # Áp dụng CLAHE lên kênh độ sáng
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])
#
# # Chuyển lại sang BGR
# result = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
#
# # Lưu ảnh
# cv2.imwrite('result.jpg', result)
