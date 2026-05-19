'''
This code finetunes the already trained SegFormer model 
to segment microscopic images of activated sludge into background, filament and flocs

the model is already pretrained on the PBM dataset and will here be finetuned using only a few images of the pilEAUte dataset

this didn't work! pileaute images need to be merged with PCM and model needs to be trained together on them
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
import time
import matplotlib.pyplot as plt
from tifffile import imread


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
        mask = imread(os.path.join(self.mask_dir, self.masks[idx]))
        mask=mask.astype(np.uint8)
        #orig_image=image

        if self.transform is not None:
            augmented = self.transform(image=np.array(image), mask=mask)
            image = augmented["image"]
            label = augmented["mask"]

        # Albumentations already returns tensors if ToTensorV2 is used
        image = image.float()
        label = label.long()

        return image, label#, orig_image

##### maybe i should make augmentations stronger -- TBD
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


## load dataset
image_dir='data/pilEAUte/finetuned_train_im_masks/images'
mask_dir='data/pilEAUte/finetuned_train_im_masks/masks'

train_dataset_full = SegmentationDataset(image_dir, mask_dir, transform=train_transform)

# ### plot augmentations 
# for idx in range(0,5):
#     image, label, orig_image = train_dataset_full[idx]

#     # Plot the images side by side
#     plt.figure(figsize=(15, 5), dpi=200)

#     # Original image subplot
#     plt.subplot(1, 3, 1)
#     plt.imshow(orig_image)
#     plt.title("Original Image")
#     plt.axis('off')  # Turn off axis labels

#     # Original image subplot
#     plt.subplot(1, 3, 2)
#     plt.imshow(image.permute(1, 2, 0))
#     plt.title("Augmented Image")
#     plt.axis('off')  # Turn off axis labels

#     # Transformed image subplot
#     plt.subplot(1, 3, 3)
#     plt.imshow(label)
#     plt.title("Transformed Image")
#     plt.axis('off')  # Turn off axis labels

#     # Show the plot
#     plt.tight_layout()
#     plt.show()


np.random.seed(25)      # for reproducibility

# try without validation (because only 8 masks available!!)
train_loader = DataLoader(train_dataset_full, batch_size=2, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
val_loader=None

# Move model to GPU
torch.cuda.set_device(3) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)

model = torch.load('outputs/trained_SegFormer.pt', map_location=device)

#define a mixed loss fct
class MixedLoss(nn.Module):
    def __init__(self, coef_ce=0.5, coef_dice=0.5, device=device):
        super().__init__()
        class_weights = torch.tensor([0.05, 0.35, 0.6], dtype=torch.float32).to(device) # low frequency -> higher weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.coef_ce = coef_ce
        self.coef_dice=coef_dice

    def forward(self, logits, labels):
        # logits: [B,C,H,W], labels: [B,H,W]
        ce_loss = self.ce(logits, labels)
        dice_loss = self.dice(logits, labels)

        loss = self.coef_ce * ce_loss  + self.coef_dice*dice_loss
        return loss, ce_loss, dice_loss

criterion = MixedLoss()

# Optimizer and LR Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7, betas=(0.9,0.999), weight_decay=0.01)

num_epochs = 50

skip_epoch_stats= False
plot_losses_path='outputs_finetunedmodel/losses.png'
save_model_path = 'outputs_finetunedmodel/finetuned_SegFormer.pt'

# --------------------------------------------------------
# Training Loop
model.to(device)
log_dict = {'train_loss_per_epoch': [], 
            'train_ce' : [], 
            'train_dice' : [],
            'val_loss_per_epoch': [],
            'val_ce' : [],
            'val_dice' : [] }
start_time = time.time()
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0  # Initialize epoch loss
    train_sum_ce_loss = 0
    train_sum_dice_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss, ce_loss, dice_loss = criterion(outputs, masks)       
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_sum_ce_loss += ce_loss.item()
        train_sum_dice_loss += dice_loss.item()

    avg_train_loss = train_loss / len(train_loader) # train loss for this epoch
    avg_ce_loss = train_sum_ce_loss / len(train_loader)
    avg_dice_loss = train_sum_dice_loss / len(train_loader)
    log_dict['train_loss_per_epoch'].append(avg_train_loss)
    log_dict['train_ce'].append(avg_ce_loss)
    log_dict['train_dice'].append(avg_dice_loss)

    avg_val_loss = float('nan') # Use NaN if no validation
    if val_loader is not None:
        model.eval()
        val_loss = 0
        val_sum_ce_loss = 0
        val_sum_dice_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss, ce_loss, dice_loss=criterion(outputs, masks)
                val_loss += loss.item()
                val_sum_ce_loss += ce_loss.item()
                val_sum_dice_loss += dice_loss.item()

        avg_val_loss= val_loss / len(val_loader) # val loss for this epoch
        avg_val_ce_loss= val_sum_ce_loss / len(val_loader) # val loss for this epoch
        avg_val_dice_loss= val_sum_dice_loss / len(val_loader) # val loss for this epoch
        log_dict['val_loss_per_epoch'].append(avg_val_loss)
        log_dict['val_ce'].append(avg_val_ce_loss)
        log_dict['val_dice'].append(avg_val_dice_loss)

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
        plt.plot(log_dict['train_ce'], '.-', label='CE train loss')
        plt.plot(log_dict['train_dice'], '.-', label='Dice train loss')

        plt.plot(log_dict['val_loss_per_epoch'], '.-', label='Total val loss')
        plt.plot(log_dict['val_ce'], '.-', label='CE val loss')
        plt.plot(log_dict['val_dice'], '.-', label='Dice val loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(plot_losses_path, dpi=300, bbox_inches='tight', pad_inches=0.1)  

    if save_model_path is not None:
        torch.save(model, save_model_path)
