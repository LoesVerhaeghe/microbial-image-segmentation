import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split, Subset
import torch
import torch.nn as nn
import time

image_dir='original_images/cropped_images'
mask_dir='train_masks/final_masks_cropped_images'

# Initialize UNet with ResNet backbone, using pretrained weights
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation=None)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

        # Get image filenames without extension
        self.image_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".JPG")]

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        name = self.image_names[idx]

        img_path = os.path.join(self.image_dir, f"{name}.JPG")
        mask_path = os.path.join(self.mask_dir, f"{name}.tif")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # single-channel for mask

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image, mask = augmented["image"], augmented["mask"]

        mask = mask.unsqueeze(0).float()  # from [H, W] -> [1, H, W]
        return image, mask
    

train_transform = A.Compose([
    A.RandomResizedCrop(size=(1024, 1024), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GridDistortion(p=0.2),
    A.ElasticTransform(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

val_transform = A.Compose([
    A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

# dataset loader
train_dataset_full = SegmentationDataset(image_dir, mask_dir, transform=train_transform)

# Val dataset without augmentations
val_dataset_full = SegmentationDataset(image_dir, mask_dir, transform=val_transform)

# Split indices
dataset_size = len(train_dataset_full)
indices = list(range(dataset_size))
split = int(0.9 * dataset_size)

train_indices = indices[:split]
val_indices = indices[split:]

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)
train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4,  num_workers=1, shuffle=False, pin_memory=True)

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice
    
criterion = DiceLoss()
    
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=100)

scaler = torch.cuda.amp.GradScaler()

torch.cuda.set_device(3) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)

num_epochs=2000
patience = 25
skip_epoch_stats=False
save_model_path=None

############ train loop

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
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        
    scheduler.step()

    avg_train_loss = train_loss / len(train_loader) # train loss for this epoch
    log_dict['train_loss_per_epoch'].append(avg_train_loss)

    avg_val_loss = float('nan') # Use NaN if no validation
    if val_loader is not None:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.cuda.amp.autocast():
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

    if not skip_epoch_stats:
        print(f'Epoch [{epoch + 1}/{num_epochs}] | Time: {((time.time() - epoch_start_time)/60):.2f} min')
        print(f'  Train Loss: Total={avg_train_loss:.4f}')
        if val_loader is not None:
            print(f'  Val Loss  : Total={avg_val_loss:.4f}')
        else:
            print()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    # if plot_losses_path is not None:
    #     plt.figure()
    #     plt.plot(log_dict['train_total_loss_per_epoch'], '.-', label='Total train loss')
    #     plt.plot(log_dict['train_mse_loss_per_epoch'], '.-', label='MSE train loss')
    #     plt.plot(log_dict['train_ssim_loss_per_epoch'], '.-', label='SSIM train loss')
    #     plt.plot(log_dict['val_total_loss_per_epoch'], '.-', label='Total val loss')
    #     plt.plot(log_dict['val_mse_loss_per_epoch'], '.-', label='MSE val loss')
    #     plt.plot(log_dict['val_ssim_loss_per_epoch'], '.-', label='SSIM val loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.savefig(f'{plot_losses_path}', dpi=300, bbox_inches='tight', pad_inches=0.1)  
    #     plt.show()

    if save_model_path is not None:
        torch.save(model, save_model_path)



### testing!!

import matplotlib.pyplot as plt

model.eval()
# Disable gradient calculation
with torch.no_grad():
    for idx in range(len(train_dataset)):
        image, mask = val_dataset[idx]
        # Add batch dimension
        image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
        mask = mask.unsqueeze(0).to(device)    # [1, 1, H, W]

        # Forward pass
        output = model(image)  # [1, 1, H, W]

        # Convert logits to probabilities
        pred_mask = torch.sigmoid(output)
        # Threshold to get binary mask
        binary_mask = (pred_mask > 0.5).float()

        # Move tensors to CPU for visualization
        img_np = image[0].permute(1,2,0).cpu().numpy()
        mask_np = mask[0,0].cpu().numpy()
        pred_np = binary_mask[0,0].cpu().numpy()

        
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = image[0].cpu().permute(1,2,0).numpy()
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)

        # Plot original, true mask, and predicted mask
        plt.figure(figsize=(8,4))
        # Overlay ground truth
        plt.subplot(1,2,1)
        plt.imshow(img_np)
        plt.imshow(mask_np, cmap='Reds', alpha=0.5)  # alpha controls transparency
        plt.title("Image + Ground Truth")
        plt.axis('off')

        # Overlay predicted mask
        plt.subplot(1,2,2)
        plt.imshow(img_np)
        plt.imshow(pred_np, cmap='Reds', alpha=0.5)
        plt.title("Image + Predicted Mask")
        plt.axis('off')

        plt.savefig(f'outputs/fig{idx}')
        # Optional: break after a few images
        if idx >= 1:
            break
