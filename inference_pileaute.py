import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
import os
from PIL import Image
import albumentations as A
from os import listdir
from albumentations.pytorch import ToTensorV2


def extract_image_paths_pileaute(path_to_folders, start_folder, end_folder, magnification=10):
    """
    Extract paths from all the images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        start_folder: start date from which images need to be extracted
        end_folder: end date until which images need to be extracted
        magnification: type of magnification (10 or 40)

    Returns:
        all_images (list): A list of all extracted images.
    """
    image_folders = sorted(listdir(path_to_folders)) 

    all_paths = []

    # Select the images from start until end date
    selected_folders = [folder for folder in image_folders if start_folder <= folder <= end_folder]
    selected_folders = sorted(selected_folders)

    # Save all paths from the selected folders
    for folder in selected_folders:
        path_to_image = f"{path_to_folders}/{folder}/basin5/{magnification}x"
        if not os.path.exists(path_to_image):  # <- skip missing folders
            continue
        images_list = sorted(listdir(path_to_image))
        for image in images_list:
            all_paths.append(f"{path_to_image}/{image}")
    return all_paths

class pilEAUteDataset(Dataset):
    def __init__(self, image_dir, start_folder, end_folder, magnification, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = extract_image_paths_pileaute(image_dir, start_folder=start_folder, end_folder=end_folder, magnification=magnification)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            augmented = self.transform(image=np.array(image))
            image = augmented["image"]

        # Albumentations already returns tensors if ToTensorV2 is used
        image = image.float()

        return image

val_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
]
    )

image_dir='data/pilEAUte/all_images'

dataset = pilEAUteDataset(image_dir, start_folder='2023-10-13', end_folder='2025-02-19', magnification=10, transform=val_transform)
data_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, drop_last=False)

torch.cuda.set_device(3) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)

save_model_path = 'outputs/trained_SegFormer.pt'
model = torch.load(save_model_path, map_location=device)

model.eval()

COLORS = {
    0: [0, 0, 0],        # background
    1: [255, 0, 0],      # class 1 - red
    2: [0, 255, 0],      # class 2 - green
}

def decode_mask(mask):
    """Convert [H, W] class mask → RGB image"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb

import random
n_samples = 30
random.seed(32)  # for reproducibility
sample_indices = random.sample(range(len(dataset)), n_samples)

with torch.no_grad():
    for idx in sample_indices:
        image = dataset[idx]

        # Add batch dimension
        image = image.unsqueeze(0).to(device)  # [1, 3, H, W]

        # Forward pass
        output = model(image)  # [1, 3, H, W]

        # Convert logits to probabilities
        probs = torch.softmax(output, dim=1)
        pred_mask = torch.argmax(probs, dim=1)[0]   # [H, W]

        # Move tensors to CPU for visualization
        img_np = image[0].permute(1,2,0).cpu().numpy()
        pred_np = pred_mask.cpu().numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)

        pred_rgb = decode_mask(pred_np)

        # Plot original, true mask, and predicted mask
        plt.figure(figsize=(12,4))
        # Overlay ground truth
        plt.subplot(1,3,1)
        plt.imshow(img_np)
        plt.title("Image")
        plt.axis('off')

        # Overlay predicted mask
        plt.subplot(1,3,2)
        plt.imshow(pred_rgb)
        plt.title("Predicted Mask")
        plt.axis('off')

        # Overlay predicted mask
        plt.subplot(1,3,3)
        plt.imshow(img_np)
        plt.imshow(pred_rgb, alpha=0.4)
        plt.title("Image+predicted mask")
        plt.axis('off')

        #plt.savefig(f'outputs/example_masks/fig{idx}')