import torch
from torchvision.models.segmentation import deeplabv3_resnet101
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLOv10
model_path = r"/yolov10/weights/best.pt"
model = YOLOv10(model_path)
# # Tải mô hình DeepLab pretrained
# model = deeplabv3_resnet101(pretrained=True)
# model.eval()  # Chế độ đánh giá
def preprocess_image(image):
    # Tăng cường độ sáng
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 40)  # Tăng độ sáng
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# Đọc ảnh
image = cv2.imread(r'/data_image/img2.jpg')
image = preprocess_image(image)

# Chuyển đổi ảnh sang không gian màu RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Tăng độ sáng bằng cách cộng giá trị cho các pixel
def increase_brightness(image, value=50):
    # Dùng numpy để cộng giá trị cho tất cả các pixel
    image_bright = cv2.convertScaleAbs(image, alpha=1, beta=value)
    return image_bright

# Tăng độ sáng ảnh
bright_image = increase_brightness(image_rgb, value=50)
# Dự đoán trên ảnh đã lọc
results = model.predict(cv2.cvtColor(bright_image, cv2.COLOR_RGB2BGR))

# Hiển thị kết quả
results[0].show()

# Hiển thị ảnh đã tăng độ sáng
# cv2.imshow('Brightened Image', cv2.cvtColor(bright_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()


# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Chuyển ảnh sang không gian màu HSV
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# # Định nghĩa các phạm vi màu sắc cho đèn giao thông
# lower_red = np.array([0, 100, 100])
# upper_red = np.array([10, 255, 255])
# lower_yellow = np.array([20, 100, 100])
# upper_yellow = np.array([40, 255, 255])
# lower_green = np.array([40, 100, 100])
# upper_green = np.array([80, 255, 255])
#
# # Tạo mặt nạ cho các màu đỏ, vàng, xanh
# mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
# mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
# mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
#
# # Kết hợp các mặt nạ
# final_mask = cv2.bitwise_or(mask_red, mask_yellow)
# final_mask = cv2.bitwise_or(final_mask, mask_green)
#
# # Áp dụng mặt nạ lên ảnh gốc
# result = cv2.bitwise_and(image, image, mask=final_mask)
#
# # Hiển thị kết quả
# cv2.imshow('Detected Lights', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# #
# # # Tiền xử lý ảnh (định dạng lại kích thước ảnh để mô hình nhận diện)
# # input_image = torch.tensor(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255
# # input_image = torch.nn.functional.interpolate(input_image, size=(640, 640), mode='bilinear', align_corners=False)
# #
# # # Dự đoán phân đoạn
# # with torch.no_grad():
# #     output = model(input_image)['out'][0]
# #     output_predictions = output.argmax(0).numpy()
# #
# # # Hiển thị kết quả phân đoạn
# # plt.imshow(output_predictions)
# # plt.show()
