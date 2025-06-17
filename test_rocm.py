import torch
import os

# Set ROCm environment variables for MI300X
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HIP_PLATFORM"] = "amd"
os.environ["HIP_COMPILER"] = "clang"
os.environ["HSA_ENABLE_SDMA"] = "0"

print("PyTorch version:", torch.__version__)
print("ROCm version:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA device:", torch.cuda.get_device_name(0))
    
    # Try a simple tensor operation
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print("Matrix multiplication successful!")
    print("Result shape:", z.shape)
else:
    print("CUDA not available") 