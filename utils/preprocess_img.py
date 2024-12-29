import cv2
def preprocess_image(image):
    # Tăng cường độ sáng
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)  # Tăng độ sáng
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)