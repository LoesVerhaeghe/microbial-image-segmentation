from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

class TestSegmentationDatasetPCM(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_np = np.array(Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB"))
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

        return image_np, label

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

    prob_map /= count_map # (3,H,W)

    return prob_map

test_image_dir='data/paper_PCM/test/images'
test_mask_dir='data/paper_PCM/test/labels'

test_dataset = TestSegmentationDatasetPCM(test_image_dir, test_mask_dir)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, drop_last=False)

torch.cuda.set_device(3) 
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_model_path = 'outputs/trained_SegFormer_v3.pt'
model = torch.load(saved_model_path, map_location=device)

model.to(device)
model.eval()

all_probs_floc = []
all_targets_floc = []

all_probs_fil = []
all_targets_fil = []

with torch.no_grad():
    for image_np, mask in test_loader:

        prob_map = predict_full_image(model, image_np[0].numpy(), device, val_transform=val_transform, tile_size=1024, overlap=64, num_classes=3)

        # 0 is background, 1 is flocs, 2 is filaments
        p_floc = prob_map[1]
        p_filament = prob_map[2]

        # ground truth
        gt_floc = (mask.squeeze(0).numpy() == 1)
        gt_filament = (mask.squeeze(0).numpy() == 2)

        all_probs_floc.append(p_floc.ravel())
        all_targets_floc.append(gt_floc.ravel())

        all_probs_fil.append(p_filament.ravel())
        all_targets_fil.append(gt_filament.ravel())

all_probs_floc = np.concatenate(all_probs_floc)
all_targets_floc = np.concatenate(all_targets_floc)

all_probs_fil = np.concatenate(all_probs_fil)
all_targets_fil = np.concatenate(all_targets_fil)

PrecisionRecallDisplay.from_predictions(
    all_targets_floc, all_probs_floc)
plt.show()

PrecisionRecallDisplay.from_predictions(
    all_targets_fil, all_probs_fil)
plt.show()


##### best threshold based on RP curve
### FLOC threshold
precision, recall, thresholds = precision_recall_curve(all_targets_floc, all_probs_floc)

f1 = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-8
)

best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]

print("floc best_threshold: ", best_threshold)
print("floc best f1: ", f1[best_idx])

### FILAMENT threshold
precision, recall, thresholds = precision_recall_curve(all_targets_fil, all_probs_fil)

f1 = 2 * precision[:-1] * recall[:-1] / (
    precision[:-1] + recall[:-1] + 1e-8
)

best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]

print("filament best_threshold: ", best_threshold)
print("filament best f1: ", f1[best_idx])




# ##### best threshold based on IoU

def find_best_iou_threshold(probs, targets):

    thresholds = np.linspace(0, 1, 50)

    best_iou = 0
    best_threshold = 0

    for t in thresholds:

        preds = probs >= t

        TP = np.logical_and(preds == 1, targets == 1).sum()
        FP = np.logical_and(preds == 1, targets == 0).sum()
        FN = np.logical_and(preds == 0, targets == 1).sum()

        iou = TP / (TP + FP + FN + 1e-8)

        if iou > best_iou:
            best_iou = iou
            best_threshold = t

    return best_threshold, best_iou

floc_threshold, floc_iou = find_best_iou_threshold(
    all_probs_floc,
    all_targets_floc
)

fil_threshold, fil_iou = find_best_iou_threshold(
    all_probs_fil,
    all_targets_fil
)

print("Floc threshold:", floc_threshold)
print("Floc IoU:", floc_iou)

print("Filament threshold:", fil_threshold)
print("Filament IoU:", fil_iou)