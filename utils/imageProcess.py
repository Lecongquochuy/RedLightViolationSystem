import cv2
import numpy as np

from yolov10.utils.preprocess_img import preprocess_image

# Đọc hình ảnh
image = cv2.imread('/mnt/data/image.png')
video_path = r"D:\KhoaLuan\yolov10\data_video\Recording-2024-08-19-214233.mp4"


cap = cv2.VideoCapture(video_path)

pixels_to_cut = 200
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = frame_width - 2 * pixels_to_cut
new_height = frame_height - 2*pixels_to_cut

while True:
    ret, frame = cap.read()
    if not ret:
        break


    cropped_frame = frame[0:new_height, pixels_to_cut:frame_width - pixels_to_cut]
    resized_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
    frame = preprocess_image(resized_frame)

    # Chuyển đổi ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Giảm độ sáng (Value) trong kênh V
    h, s, v = cv2.split(hsv)
    v = cv2.inRange(v, 150, 255)  # Giảm độ sáng từ các vùng rất sáng
    v = cv2.GaussianBlur(v, (11, 11), 0)  # Làm mờ để giảm sự gắt của vùng sáng

    # Gộp lại kênh HSV
    hsv_reduced = cv2.merge((h, s, v))
    reduced_image = cv2.cvtColor(hsv_reduced, cv2.COLOR_HSV2BGR)

    # Áp dụng bộ lọc giảm nhiễu
    denoised_image = cv2.fastNlMeansDenoisingColored(reduced_image, None, 10, 10, 7, 21)

    # Hiển thị và lưu hình ảnh sau khi giảm lóa sáng
    # cv2.imshow('Original Image', frame)
    # cv2.imshow('Reduced Highlights', reduced_image)
    cv2.imshow('Denoised Image', denoised_image)
    # cv2.imwrite('/mnt/data/reduced_highlight_denoised_image.png', denoised_image)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
cap.release()
cv2.destroyAllWindows()
