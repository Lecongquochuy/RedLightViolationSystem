import torch

# Kiểm tra GPU có sẵn
print("PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected.")
