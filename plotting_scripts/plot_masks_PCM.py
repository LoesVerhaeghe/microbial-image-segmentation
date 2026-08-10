"""
Generate and plot some masks for PCM images and save them for visualization
Also save some masks to later finetune the model on them
"""


import shutil

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
import os
from PIL import Image
import albumentations as A
from os import listdir
from albumentations.pytorch import ToTensorV2
from tifffile import imread


val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})


def predict_full_image(model, image_np, device, val_transform=val_transform, tile_size=1024, overlap=64, num_classes=3):
    model.eval()

    stride = tile_size - overlap
    H, W, _ = image_np.shape

    prob_map = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            tile = image_np[y:y+tile_size, x:x+tile_size]

            h_tile, w_tile = tile.shape[:2]

            # pad if at border
            if h_tile < tile_size or w_tile < tile_size:
                pad_img = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                pad_img[:h_tile, :w_tile] = tile
                tile = pad_img

            # transform
            augmented = val_transform(image=tile)
            tile_tensor = augmented["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tile_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()

            probs = probs[:, :h_tile, :w_tile]

            prob_map[:, y:y+h_tile, x:x+w_tile] += probs
            count_map[y:y+h_tile, x:x+w_tile] += 1

    prob_map /= count_map # (3,H,W)
    #final_mask = np.argmax(prob_map, axis=0)
    # -----------------------------------
    # threshold-based classification
    # -----------------------------------

    bg_prob = prob_map[0]
    floc_prob = prob_map[1]
    filament_prob = prob_map[2]

    # start with background
    final_mask = np.zeros((H, W), dtype=np.uint8)

    floc_pixels = floc_prob >= 0.79279387
    filament_pixels = filament_prob >= 0.6926654

    # assign flocs
    final_mask[floc_pixels] = 1

    # assign filaments
    final_mask[filament_pixels] = 2

    # resolve pixels where both pass
    both = floc_pixels & filament_pixels

    final_mask[both & (floc_prob >= filament_prob)] = 1
    final_mask[both & (filament_prob > floc_prob)] = 2

    return prob_map, final_mask


COLORS = {
    0: [0, 0, 0],        # background
    1: [255, 0, 0],      # class 1 - red
    2: [0, 255, 0],      # class 2 - green
}

def decode_mask(mask, COLORS):
    """Convert [H, W] class mask → RGB image"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb

mask_dir='data/paper_PCM/test/labels'
image_dir='data/paper_PCM/test/images'
images= listdir(image_dir)

torch.cuda.set_device(3) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)

save_model_path = 'outputs/trained_SegFormer.pt'
model = torch.load(save_model_path, map_location=device)

model.eval()

indices = np.random.choice(len(images), size=20, replace=False)

#plot some infered mask for visual evaluation
with torch.no_grad():
    for idx in indices:
        image_path = os.path.join(image_dir, images[idx])
        image_np = np.array(Image.open(image_path).convert("RGB"))
        real_mask_path = os.path.join(mask_dir, images[idx])
        ground_truth = np.array(Image.open(real_mask_path).convert("RGB"))
        ground_truth[(ground_truth == [128, 0, 0]).all(axis=-1)] = [255, 0, 0]
        ground_truth[(ground_truth == [0, 128, 0]).all(axis=-1)] = [0, 255, 0]

        pred_prob, mask_np = predict_full_image(model, image_np, device, val_transform=val_transform, tile_size=1024, overlap=64, num_classes=3)
        pred_rgb = decode_mask(mask_np, COLORS)

        # Plot original, predicted mask and overlay
        plt.figure(figsize=(12,4), dpi=500)
        # Overlay ground truth
        plt.subplot(1,3,1)
        plt.imshow(image_np)
        plt.title("Image")
        plt.axis('off')

        # Overlay predicted mask
        plt.subplot(1,3,2)
        plt.imshow(pred_rgb)
        plt.title("Predicted Mask")
        plt.axis('off')

        # Overlay ground truth mask
        plt.subplot(1,3,3)
        plt.imshow(ground_truth)
        plt.title("ground truth mask")
        plt.axis('off')

        plt.savefig(f'outputs/example_masks_PCM/fig{idx}')

