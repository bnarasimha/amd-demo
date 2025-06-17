from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import torch
import os
import sys
import trimesh

# Set ROCm environment variables for MI300X
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"  # MI300X uses GFX11
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HIP_PLATFORM"] = "amd"
os.environ["HIP_COMPILER"] = "clang"
os.environ["HSA_ENABLE_SDMA"] = "0"  # Disable SDMA for better compatibility

# Print debug information
print("PyTorch version:", torch.__version__)
print("ROCm version:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available())

# Determine device
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("CUDA not available, using CPU")

try:
    # Initialize pipeline
    print("Loading pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    print("Moving pipeline to device:", device)
    pipeline.to(device)
    
    # Generate mesh
    print("Generating mesh...")
    mesh = pipeline(image='reference_images/3d.png')[0]
    
    # Paint the mesh
    print("Painting mesh...")
    pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    
    print("Moving pipeline to device:", device)
    pipeline.to(device)
    
    print("Generating painted mesh...")
    mesh = pipeline(mesh, image='reference_images/3d.png')
    
    # Save the mesh
    print("Saving mesh...")
    trimesh.exchange.export.export_mesh(mesh, "output/3d.obj")
    print("Done!")

except Exception as e:
    print("Error occurred:", str(e), file=sys.stderr)
    sys.exit(1)

