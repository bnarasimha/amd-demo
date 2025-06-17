import os
import numpy as np
from PIL import Image
import nibabel as nib

# Path to your image folder
img_folder = 'reference_images'
img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

# Load first image to get target dimensions
first_img = Image.open(os.path.join(img_folder, img_files[0])).convert('L')
target_size = first_img.size  # (width, height)

# Load images and stack into a 3D numpy array
img_stack = []
for fname in img_files:
    img = Image.open(os.path.join(img_folder, fname)).convert('L')  # 'L' for grayscale
    # Resize image to match the first image's dimensions
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_stack.append(np.array(img))

img_3d = np.stack(img_stack, axis=-1)  # shape: (height, width, num_slices)

# Create NIfTI image
nii_img = nib.Nifti1Image(img_3d, affine=np.eye(4))

# Save as .nii file
nib.save(nii_img, 'output/output.nii.gz')

print(f"NIfTI file saved as output.nii.gz")
print(f"Volume shape: {img_3d.shape}")