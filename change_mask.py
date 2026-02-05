from PIL import Image
import torch
import numpy as np

mask_path='data/paper_PCM/labels/1231cout19.png'

mask = Image.open(mask_path).convert("RGB")
mask = np.array(mask)

label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)

# red → class 1
label[(mask[:, :, 0] == 128) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)] = 1

# green → class 2
label[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 128) & (mask[:, :, 2] == 0)] = 2

label = torch.from_numpy(label)
