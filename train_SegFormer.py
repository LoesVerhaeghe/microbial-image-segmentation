'''
This code finetunes the SegFormer model 
to segment microscopic images of activated sludge into background, filament and flocs
it does use the PBM and pilEAUte segmented images
'''

import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, ConcatDataset
import segmentation_models_pytorch as smp
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from tifffile import imread
import copy 

class SegmentationDatasetPCM(Dataset):
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

        return image, label

class SegmentationDatasetPilEAUte(Dataset):
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

##  horizontal and vertical flipping and rotation only to train dataset
train_transform_PCM = A.Compose([
    # ---- scale robustness ----
    A.OneOf([
        A.RandomScale(scale_limit=(-0.3, 0)),
        A.RandomScale(scale_limit=(0.0, 0.3)),
    ], p=0.7),

    A.PadIfNeeded(min_height=1024, min_width=1024),
    A.RandomCrop(1024, 1024),

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
    A.Downscale(scale_range=[0.8,0.95], p=0.4),

    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

train_transform_pilEAUte = A.Compose([
    # ---- scale robustness ----
    A.OneOf([
        A.RandomScale(scale_limit=(-0.3, 0)),
        A.RandomScale(scale_limit=(0.0, 0.3)),
    ], p=0.7),

    A.PadIfNeeded(min_height=1024, min_width=1024),
    A.RandomCrop(1024, 1024),

    # ---- geometry ----
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.8,1.2), rotate=(20), shear=(-10,10), p=0.5), # larger scale

    # ---- microscopy realism ----
    A.OneOf([
        A.GaussianBlur(blur_limit=5),
        A.MotionBlur(blur_limit=5), # larger blur
    ], p=0.5),

    A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), p=0.5), # larger contrast
    A.RandomGamma(p=0.3),

    # ---- slight resolution degradation ----
    A.Downscale(scale_range=[0.8,0.95], p=0.5), # larger scale

    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

val_transform = A.Compose([
    A.PadIfNeeded(min_height=1024, min_width=1024),
    A.CenterCrop(1024, 1024),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})


## load dataset PCM
image_dir_PCM='data/paper_PCM/train/images'
mask_dir_PCM='data/paper_PCM/train/labels'

train_dataset_full_PCM = SegmentationDatasetPCM(image_dir_PCM, mask_dir_PCM, transform=train_transform_PCM)
val_dataset_PCM = SegmentationDatasetPCM(image_dir_PCM, mask_dir_PCM, transform=val_transform)

dataset_size_PCM = len(train_dataset_full_PCM)
indices_PCM = list(range(dataset_size_PCM))

np.random.seed(25)      # for reproducibility
np.random.shuffle(indices_PCM)

split_PCM = int(0.90 * dataset_size_PCM)
train_indices_PCM = indices_PCM[:split_PCM]
val_indices_PCM = indices_PCM[split_PCM:]

train_dataset_PCM = Subset(train_dataset_full_PCM, train_indices_PCM)
val_dataset_PCM = Subset(val_dataset_PCM, val_indices_PCM)

# load dataset pilEAUte
image_dir_pilEAUte='data/pilEAUte/finetuned_train_im_masks/images'
mask_dir_pilEAUte='data/pilEAUte/finetuned_train_im_masks/masks'

train_dataset_pilEAUte = SegmentationDatasetPilEAUte(image_dir_pilEAUte, mask_dir_pilEAUte, transform=train_transform_pilEAUte)
val_dataset_pilEAUte = SegmentationDatasetPilEAUte(image_dir_pilEAUte, mask_dir_pilEAUte, transform=val_transform)

dataset_size_pilEAUte = len(train_dataset_pilEAUte)
indices_pilEAUte = list(range(dataset_size_pilEAUte))

split_pilEAUte = int(0.80 * dataset_size_pilEAUte)
train_indices_pilEAUte = indices_pilEAUte[:split_pilEAUte]
val_indices_pilEAUte = indices_pilEAUte[split_pilEAUte:]

train_dataset_pilEAUte = Subset(train_dataset_pilEAUte, train_indices_pilEAUte)
val_dataset_pilEAUte = Subset(val_dataset_pilEAUte, val_indices_pilEAUte)

### combine dataset
combined_train_dataset = ConcatDataset([train_dataset_PCM, train_dataset_pilEAUte])
combined_val_dataset = ConcatDataset([val_dataset_PCM, val_dataset_pilEAUte])

pcm_weights = [1.0] * len(train_dataset_PCM)
pileaute_weights = [10.0] * len(train_dataset_pilEAUte)  # oversample factor

weights = pcm_weights + pileaute_weights 

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=2*len(weights),  
    replacement=True # because we only crop out a small part of the image the same image can be used twice as sample
)

train_loader = DataLoader(combined_train_dataset, batch_size=8, num_workers=2, shuffle=False, sampler=sampler, pin_memory=True, drop_last=True)
val_loader = DataLoader(combined_val_dataset, batch_size=8,  num_workers=2, shuffle=False, pin_memory=True)


# Move model to GPU
torch.cuda.set_device(0) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(25)


## load model
num_classes = 3
model = smp.Segformer(
    encoder_name="mit_b1",             
    encoder_weights='imagenet',   
    decoder_segmentation_channels=128, # channels in decoder, can tune
    in_channels=3,                      
    classes=num_classes,               
    activation=None,                   
    upsampling=4                      # final upsampling factor
)

#define a mixed loss fct
class MixedLoss(nn.Module):
    def __init__(self, coef_ce=0.4, coef_dice=0.6, device=device):
        super().__init__()
        class_weights = torch.tensor([0.05, 0.30, 0.65], dtype=torch.float32).to(device) # low frequency -> higher weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights) # quantifies how far off the model's predictions are from the true labels, weighted to account for class imbalance
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=[1,2]) # area of overlap / area of union
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
# beta1: smooth direction of learning, beta2: smooth scaling of step size, 
# weight decay: regularization to prevent overfitting, penalize large weights
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00024, betas=(0.9,0.999), weight_decay=0.01) 

num_epochs = 300

scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer,
    total_iters=num_epochs,   
    power=2.0,
)

patience = 20
skip_epoch_stats= False
plot_losses_path='outputs/losses_v3.png'
save_model_path = 'outputs/trained_SegFormer_v3.pt'

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
best_model_weights = copy.deepcopy(model.state_dict())

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
            # save best model weights
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                # restore best model
                model.load_state_dict(best_model_weights)
                break  # Stop training

    scheduler.step()

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
    plt.savefig(f'{plot_losses_path}', dpi=300, bbox_inches='tight', pad_inches=0.1)  

if save_model_path is not None:
    torch.save(model, save_model_path)

