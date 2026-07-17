'''
This code plots the image + mask augmentations that were applied to 
train the SegFormer to check if the augmentations are correct and realistic
'''
 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("RGB")
        mask = np.array(mask)

        orig_image= image.copy()

        # --- COLOR → CLASS MAP ---
        label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        label[(mask[:, :, 0] == 128) &
              (mask[:, :, 1] == 0) &
              (mask[:, :, 2] == 0)] = 1 # red is floc
        label[(mask[:, :, 0] == 0) &
              (mask[:, :, 1] == 128) &
              (mask[:, :, 2] == 0)] = 2 # green is filament

        if self.transform is not None:
            augmented = self.transform(image=np.array(image), mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # Albumentations already returns tensors if ToTensorV2 is used
        image = image.float()
        label = label.long()

        return image, label, orig_image
    

##  horizontal and vertical flipping and rotation only to train dataset
train_transform = A.Compose([
    # ---- scale robustness ----
    A.OneOf([
        A.RandomScale(scale_limit=(-0.6, -0.2)),
        A.RandomScale(scale_limit=(0.0, 0.3)),
    ], p=0.7),

    A.PadIfNeeded(min_height=700, min_width=700),
    A.RandomCrop(512, 512),

    # ---- geometry ----
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9,1.1), rotate=(-10,10), shear=(-5,5), p=0.3),

    # ---- microscopy realism ----
    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3),
    ], p=0.4),

    A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), p=0.4),
    A.RandomGamma(p=0.3),

    # ---- slight resolution degradation ----
    A.Downscale(scale_range=[0.75,0.9], p=0.4),

    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

val_transform = A.Compose([
    A.CenterCrop(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})


image_dir='data/paper_PCM/train/images'
mask_dir='data/paper_PCM/train/labels'


train_dataset_full = SegmentationDataset(image_dir, mask_dir, transform=train_transform)
for idx in range(0,10):
    image, label, orig_image = train_dataset_full[idx]

    # Plot the images side by side
    plt.figure(figsize=(15, 5), dpi=200)

    # Original image subplot
    plt.subplot(1, 3, 1)
    plt.imshow(orig_image)
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis labels

    # Original image subplot
    plt.subplot(1, 3, 2)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Augmented Image")
    plt.axis('off')  # Turn off axis labels

    # Transformed image subplot
    plt.subplot(1, 3, 3)
    plt.imshow(label)
    plt.title("Transformed Image")
    plt.axis('off')  # Turn off axis labels

    # Show the plot
    plt.tight_layout()
    plt.show()