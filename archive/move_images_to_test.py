import os
import random
import shutil

# Paths
image_dir = "data/paper_PCM/train/images"
mask_dir = "data/paper_PCM//train/labels"

test_image_dir = "data/paper_PCM/test/images"
test_mask_dir = "data/paper_PCM/test/labels"

# Make sure test directories exist
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)

# List all images in train folder
all_images = os.listdir(image_dir)

# Choose 50 random images
random.seed(42)  # for reproducibility
test_images = random.sample(all_images, 50)

# Move the selected images and their masks
for img_name in test_images:
    # Move image
    shutil.move(os.path.join(image_dir, img_name),
                os.path.join(test_image_dir, img_name))
    
    # Move corresponding mask
    mask_path = os.path.join(mask_dir, img_name)  # same filename
    if os.path.exists(mask_path):
        shutil.move(mask_path, os.path.join(test_mask_dir, img_name))
    else:
        print(f"Warning: mask for {img_name} not found!")

print(f"Moved {len(test_images)} images and masks to the test dataset.")
