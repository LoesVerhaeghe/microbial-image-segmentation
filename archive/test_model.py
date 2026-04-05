import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

        # --- COLOR → CLASS MAP ---
        label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        label[(mask[:, :, 0] == 128) &
              (mask[:, :, 1] == 0) &
              (mask[:, :, 2] == 0)] = 1
        label[(mask[:, :, 0] == 0) &
              (mask[:, :, 1] == 128) &
              (mask[:, :, 2] == 0)] = 2

        if self.transform is not None:
            augmented = self.transform(image=np.array(image), mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # Albumentations already returns tensors if ToTensorV2 is used
        image = image.float()
        label = label.long()

        return image, label

val_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})


test_image_dir='data/paper_PCM/test/images'
test_mask_dir='data/paper_PCM/test/labels'

test_dataset = SegmentationDataset(test_image_dir, test_mask_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, drop_last=False)

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

with torch.no_grad():
    for idx in range(len(test_dataset)):
        image, mask = test_dataset[idx]

        # Add batch dimension
        image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
        mask = mask.to(device)    # [H, W]

        # Forward pass
        output = model(image)  # [1, 3, H, W]

        # Convert logits to probabilities
        probs = torch.softmax(output, dim=1)
        pred_mask = torch.argmax(probs, dim=1)[0]   # [H, W]

        # Move tensors to CPU for visualization
        img_np = image[0].permute(1,2,0).cpu().numpy()
        mask_np = mask.cpu().numpy()
        pred_np = pred_mask.cpu().numpy()

        
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)

        mask_rgb   = decode_mask(mask_np)
        pred_rgb = decode_mask(pred_np)

        # Plot original, true mask, and predicted mask
        plt.figure(figsize=(12,4))
        # Overlay ground truth
        plt.subplot(1,3,1)
        plt.imshow(img_np)
        plt.title("Image")
        plt.axis('off')

        # # Overlay predicted mask
        # plt.subplot(1,3,2)
        # plt.imshow(mask_rgb)
        # plt.title("Ground truth")
        # plt.axis('off')

        # Overlay predicted mask
        plt.subplot(1,3,3)
        plt.imshow(pred_rgb)
        plt.title("Predicted Mask")
        plt.axis('off')

        #plt.savefig(f'outputs/example_masks/fig{idx}')