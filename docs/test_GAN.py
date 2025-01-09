import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Định nghĩa Generator class (phải phù hợp với cấu trúc model Pix2Pix đã huấn luyện)
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Định nghĩa cấu trúc của Generator theo Pix2Pix GAN
        # Thay thế bằng kiến trúc phù hợp đã được sử dụng trong huấn luyện

    def forward(self, x):
        return x

# Tải trọng số huấn luyện của Generator
G = Generator()
G.load_state_dict(torch.load("latest_net_G.pth", map_location=torch.device('cpu')))
G.eval()

# Hàm tiền xử lý ảnh đầu vào
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize ảnh về kích thước 256x256
        transforms.ToTensor(),         # Chuyển ảnh sang tensor
        transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa ảnh về [-1, 1]
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Thêm batch dimension

# Đọc ảnh biển số xe đầu vào
input_image_path = "path_to_license_plate_image.jpg"  # Thay bằng đường dẫn thực tế
input_image = preprocess_image(input_image_path)

# Tạo ảnh đầu ra bằng Generator
with torch.no_grad():
    output_image = G(input_image)

# Hiển thị kết quả bằng imshow
input_image_display = input_image.squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5
output_image_display = output_image.squeeze(0).permute(1, 2, 0).detach().numpy() * 0.5 + 0.5

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(input_image_display)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image_display)
plt.title('Output Image')
plt.axis('off')

plt.show()
