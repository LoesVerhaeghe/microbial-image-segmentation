'''
This code finetunes the SegFormer model from segmentation_models_pytorch. 
The architecture and training choices (e.g. loss function) are based on this github repo: https://github.com/SYUCT-InfoEng/PCM_data/tree/main
Some adaptation were made to their approach (explained in comments)
'''

import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset
import segmentation_models_pytorch as smp
from PIL import Image
import os
import lovasz_losses as L
import time
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

##  horizontal and vertical flipping and rotation only to train dataset
# Normalisation step was not specified in PCM approach, but added here
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
        A.GaussianBlur(blur_limit=5),
        A.MotionBlur(blur_limit=3),
    ], p=0.4),

    A.RandomBrightnessContrast(p=0.4),
    A.RandomGamma(p=0.3),

    # ---- slight resolution degradation ----
    A.Downscale(scale_range=[0.4,0.8], p=0.4),

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


## load dataset
image_dir='data/paper_PCM/train/images'
mask_dir='data/paper_PCM/train/labels'

train_dataset_full = SegmentationDataset(image_dir, mask_dir, transform=train_transform)
val_dataset_full = SegmentationDataset(image_dir, mask_dir, transform=val_transform)

dataset_size = len(train_dataset_full)
indices = list(range(dataset_size))

np.random.seed(25)      # for reproducibility
np.random.shuffle(indices)

split = int(0.90 * dataset_size)
train_indices = indices[:split]
val_indices = indices[split:]

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8,  num_workers=2, shuffle=False, pin_memory=True)

## load model
num_classes = 3
model = smp.Segformer(
    encoder_name="mit_b3",             # the backbone: mit_b3 = SegFormer B3
    encoder_weights='imagenet',   
    decoder_segmentation_channels=512, # channels in decoder, can tune
    in_channels=3,                      
    classes=num_classes,               
    activation=None,                   
    upsampling=4                      # final upsampling factor
)

# Move model to GPU
torch.cuda.set_device(3) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)

# The multiclass lovasz_softmax expect class probabilities (the maximum scoring category is predicted). 
# First use a Softmax layer on the unnormalized scores.
class MixedLoss(nn.Module):
    def __init__(self, coef_ce=0.3, coef_lovasz=0.4, coef_dice=0.3, device=device):
        super().__init__()
        class_weights = torch.tensor([0.15, 0.35, 0.5], dtype=torch.float32).to(device)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.coef_ce = coef_ce
        self.coef_lovasz = coef_lovasz
        self.coef_dice=coef_dice

    def forward(self, logits, labels):
        # logits: [B,C,H,W], labels: [B,H,W]
        ce_loss = self.ce(logits, labels)
        probs = torch.softmax(logits, dim=1)
        lovasz_loss = L.lovasz_softmax(probs, labels)  
        dice_loss = self.dice(logits, labels)
        return self.coef_ce * ce_loss + self.coef_lovasz * lovasz_loss + self.coef_dice*dice_loss

criterion = MixedLoss()

# Optimizer and LR Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, betas=(0.9,0.999), weight_decay=0.01)

num_epochs = 50

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=5
)

patience = 5
skip_epoch_stats= False
plot_losses_path='outputs/losses'
save_model_path = 'outputs/trained_SegFormer.pt'

# --------------------------------------------------------
# Training Loop
model.to(device)
log_dict = {'train_loss_per_epoch': [], 'val_loss_per_epoch': []}
start_time = time.time()
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0  # Initialize epoch loss
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)       
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader) # train loss for this epoch
    log_dict['train_loss_per_epoch'].append(avg_train_loss)

    avg_val_loss = float('nan') # Use NaN if no validation
    if val_loader is not None:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss=criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss= val_loss / len(val_loader) # val loss for this epoch
        log_dict['val_loss_per_epoch'].append(avg_val_loss)

        if avg_val_loss  < best_val_loss:
            best_val_loss = avg_val_loss 
            patience_counter = 0  # Reset the counter when improvement occurs
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                break  # Stop training
        # update learning rate using scheduler
        scheduler.step(val_loss)

    if not skip_epoch_stats:
        print(f'Epoch [{epoch + 1}/{num_epochs}] | Time: {((time.time() - epoch_start_time)/60):.2f} min')
        print(f'  Train Loss: Total={avg_train_loss:.4f}')
        if val_loader is not None:
            print(f'  Val Loss  : Total={avg_val_loss:.4f}')
        else:
            print()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    if plot_losses_path is not None:
        plt.figure()
        plt.plot(log_dict['train_loss_per_epoch'], '.-', label='Total train loss')
        plt.plot(log_dict['val_loss_per_epoch'], '.-', label='Total val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{plot_losses_path}', dpi=300, bbox_inches='tight', pad_inches=0.1)  

    if save_model_path is not None:
        torch.save(model, save_model_path)


### testing

test_image_dir='data/paper_PCM/test/images'
test_mask_dir='data/paper_PCM/test/labels'

test_dataset = SegmentationDataset(test_image_dir, test_mask_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, drop_last=False)

model.eval()

COLORS = {
    0: [0, 0, 0],        # background
    1: [255, 0, 0],      # class 1 - red
    2: [0, 255, 0],      # class 2 - green
}

#### plot some masks for evaluation

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

        # Overlay predicted mask
        plt.subplot(1,3,2)
        plt.imshow(mask_rgb)
        plt.title("Ground truth")
        plt.axis('off')

        # Overlay predicted mask
        plt.subplot(1,3,3)
        plt.imshow(pred_rgb)
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.savefig(f'outputs/example_masks/fig{idx}', dpi=300)
        plt.close()
        # Optional: break after a few images
        if idx >= 10:
            break