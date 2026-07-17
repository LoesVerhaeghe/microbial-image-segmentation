"""
for already generated masks, calculate the properties of the flocs and filaments in the mask and plot the results
"""


# 0=background, 1=floc, 2=filament
import numpy as np
from skimage import measure, morphology
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
import os

all_im_names=[name for name in os.listdir("outputs/masks_pileaute") if name.endswith('.png')]

for image_name in all_im_names[:20]:
    image = Image.open(f'outputs/masks_pileaute/{image_name}').convert("RGB")
    idx=image_name.split('_')[0]
    mask=np.load(f'outputs/masks_pileaute/{idx}_mask.npy')

    floc_mask = (mask == 1)
    filament_mask = (mask == 2)

    ### calculate floc properties:

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

    ### calculate filament properties:

    skeleton = skeletonize(filament_mask)

    # # display results
    # fig = plt.figure(figsize=(8, 4))
    # plt.imshow(skeleton, cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

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

    #### plot the results


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

    pred_rgb = decode_mask(mask, COLORS)

    # Plot original, predicted mask and overlay

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=500)

    # Left: image + mask
    axes[0].imshow(image)
    axes[0].imshow(pred_rgb, alpha=0.25)
    axes[0].axis('off')

    #  Right: metrics text
    axes[1].axis('off')

    text = "\n".join([
        f"Num flocs: {metrics['num_flocs']}", 
        f"Mean floc area: {metrics['mean_floc_area']:.2f}", 
        f"Mean circularity: {metrics['mean_circularity']:.2f}", 
        f"Total filament length: {metrics['total_filament_length']:.2f}", 
        f"Mean filament length: {metrics['mean_filament_length']:.2f}", 
        f"Filament/floc: {metrics['filament_to_floc_ratio']:.2f}", 
    ])

    axes[1].text(0, 1, text, fontsize=15, va='top')

    plt.tight_layout()
    plt.show()

