from torch.utils.data import Dataset, DataLoader
from os import listdir
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from skimage import measure, morphology
from skimage.morphology import skeletonize

### define functions
    
val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})


def predict_full_image(model, image_np, device, val_transform=val_transform, tile_size=512, overlap=128, num_classes=3):
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

    prob_map /= count_map
    final_mask = np.argmax(prob_map, axis=0)

    return final_mask

### read dataset

folder_dir='data/pilEAUte/all_images/2024-02-06/basin5/10x'
all_image_paths=[]
for image in listdir(folder_dir):
    image_path=f'{folder_dir}/{image}'
    all_image_paths.append(image_path)

device = torch.device("cpu")
model_path = 'outputs/trained_SegFormer.pt'
model = torch.load(model_path, map_location=device)
model.eval()

### segment all images

all_masks=[]
with torch.no_grad():
    for image_path in all_image_paths:
        image = np.array(Image.open(image_path).convert("RGB"))

        pred_np = predict_full_image(model, 
                                     image, 
                                     device, 
                                     val_transform=val_transform, 
                                     tile_size=512, 
                                     overlap=128, 
                                     num_classes=3)
        all_masks.append(pred_np)


### calculate properties per segmented image
all_metrics=[]
for mask in all_masks:
    floc_mask = (mask == 1)
    filament_mask = (mask == 2)

    # calculate floc properties:
    floc_mask = morphology.remove_small_objects(floc_mask, min_size=50) 
    labeled_flocs = measure.label(floc_mask)   #Label connected regions of an integer array
    floc_regions = measure.regionprops(labeled_flocs) #Measure properties of labeled image regions

    total_floc_area = np.sum(floc_mask)

    areas = []
    circularities = []
    aspect_ratios = []

    for r in floc_regions: # for every floc
        if r.perimeter_crofton  <= 0:
            continue
        area = r.area #number of pixels of the region scaled by pixel area
        perimeter = r.perimeter_crofton   #approximates the contour as a line through the centers of border pixels using a 4-connectivity.

        circularity = 4 * np.pi * area / (perimeter ** 2) #how closely a shape matches a perfect circle

        areas.append(area)
        circularities.append(circularity)

    # calculate filament properties:

    skeleton = skeletonize(filament_mask)

    total_filament_length = np.sum(skeleton)
    total_filament_area = np.sum(filament_mask)

    labeled_filaments = measure.label(skeleton)
    filament_regions = measure.regionprops(labeled_filaments)
    filament_lengths = []
    for r in filament_regions:
        length = r.area  # number of skeleton pixels
        filament_lengths.append(length)

    ## summarize all metrics
    image_area = mask.size

    metrics = {
        "num_flocs": len(areas),
        "mean_floc_area": np.mean(areas) if areas else 0,
        "mean_circularity": np.mean(circularities) if circularities else 0,
        "total_filament_length": total_filament_length,
        "mean_filament_length": np.mean(filament_lengths) if filament_lengths else 0,
        "filament_to_floc_ratio": total_filament_area / total_floc_area if total_floc_area > 0 else 0,
        "floc_area_fraction": total_floc_area / image_area,
    }
    all_metrics.append(metrics)


# calculate average properties
import pandas as pd
df=pd.DataFrame(all_metrics)