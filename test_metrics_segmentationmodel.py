
# IoU
# mIoU
# accuracy
# precision
# recall
# f1



# per class: IoU, precission, recall, f1, AUC

# average: mIoU, accuracy, FLOPs(G), Params(M), inference time/s


import os
import time
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from thop import profile


#############################################
# Dataset
#############################################

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


val_transform = A.Compose([
    A.CenterCrop(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})


test_image_dir='data/paper_PCM/test/images'
test_mask_dir='data/paper_PCM/test/labels'

test_dataset = SegmentationDatasetPCM(test_image_dir, test_mask_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, drop_last=False)

#############################################
# Metric accumulator
#############################################

NUM_CLASSES = 3
# 0 is background, 1 is flocs, 2 is filaments

confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

# Rows = true classes
# Columns = predicted classes

# This allows us to calculate:
# TP = true positives
# FP = false positives
# FN = false negatives

all_probs = []
all_labels = []

#############################################
# Evaluation
#############################################

torch.cuda.set_device(2) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_model_path = 'outputs/trained_SegFormer.pt'
model = torch.load(saved_model_path, map_location=device)

model.to(device)
model.eval()

times = []

with torch.no_grad():
    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        start = time.perf_counter()

        outputs = model(images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)[0]

        #####################################
        # Confusion matrix
        #####################################

        gt = masks.cpu().numpy().flatten()
        pred = preds.cpu().numpy().flatten()

        confmat += confusion_matrix(
            gt,
            pred,
            labels=np.arange(NUM_CLASSES)
        )

        #####################################
        # AUC
        #####################################
        # (batch, classes, H, W) -> (number_of_pixels, classes)
        all_probs.append(
            probs.permute(0,2,3,1) 
            .reshape(-1, NUM_CLASSES)
            .cpu()
            .numpy()
        )

        all_labels.append(gt)


#############################################
# Per-class metrics
#############################################

ious = []

print("\nPer-class metrics\n")

for c in range(NUM_CLASSES):

    TP = confmat[c,c]
    FP = confmat[:,c].sum() - TP # Everything predicted as class c, but actually another class
    FN = confmat[c,:].sum() - TP # Everything actually class c, but predicted as another class

    
    iou = TP / (TP + FP + FN + 1e-8) # overlap between prediction and ground truth
    precision = TP / (TP + FP + 1e-8) # Of all pixels predicted as this class: how many were correct?
    recall = TP / (TP + FN + 1e-8) # Of all pixels predicted as this class: how many did we find?
    f1 = 2 * precision * recall / (precision + recall + 1e-8) # Harmonic mean of precision and recall

    ious.append(iou)

    print(f"Class {c}")
    print(f"IoU       : {iou:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print()


#############################################
# AUC
#############################################

probs = np.concatenate(all_probs)
labels = np.concatenate(all_labels)

labels_onehot = np.eye(NUM_CLASSES)[labels]

print("AUC")

for c in range(NUM_CLASSES):

    auc = roc_auc_score(
        labels_onehot[:,c],
        probs[:,c]
    )

    print(f"Class {c}: {auc:.4f}")


#############################################
# Overall metrics
#############################################

accuracy = np.trace(confmat) / confmat.sum()
miou = np.mean(ious)

print("\nOverall metrics")
print("----------------")
print(f"mIoU              : {miou:.4f}")
print(f"Pixel Accuracy    : {accuracy:.4f}")
print(f"Inference time    : {np.mean(times):.5f} s/image")


#############################################
# FLOPs & Params
#############################################

dummy = torch.randn(1,3,512,512).to(device)

flops, params = profile(
    model,
    inputs=(dummy,),
    verbose=False
)

print(f"FLOPs : {flops/1e9:.2f} G")
print(f"Params: {params/1e6:.2f} M")