import torch
from diffusers import StableDiffusionXLPipeline
import os

# Disable NNPACK
os.environ['USE_NNPACK'] = '0'

# Set device to AMD GPU (ROCm)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline with additional configuration
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,  # Use half precision for better memory efficiency
    use_safetensors=True,
    variant="fp16"
)
pipe = pipe.to(device)

# Generate the image
prompt = "CT scan image of the human chest, highly detailed, medical imaging"
image = pipe(prompt).images[0]

# Save the image
image.save("ct_scan_chest.png")