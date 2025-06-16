# Add these lines to the top of your script (before importing torch)
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCM_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import torch
from diffusers import DiffusionPipeline
import os

# Disable NNPACK
os.environ['USE_NNPACK'] = '0'

# Force ROCm device
os.environ['HIP_VISIBLE_DEVICES'] = '0'
device = "cuda"  # For AMD ROCm, we use 'cuda' as the device name

# Load the pipeline with additional configuration
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    use_safetensors=True
)

# Move to GPU (use default float32 for ROCm compatibility)
pipe = pipe.to(device)

# Enable memory efficient attention
pipe.enable_attention_slicing()

# Generate the image
prompt = "CT scan image of the human chest, no other body parts, highly detailed, medical imaging"
image = pipe(prompt).images[0]

# Save the image
image.save("ct_scan_chest_flux.png")

print(torch.version.hip)  # Should print ROCm version if ROCm is enabled
print(torch.cuda.is_available())
print(torch.cuda.device_count())

