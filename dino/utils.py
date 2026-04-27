import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import v2


def apply_clahe_cv2_dino(img):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()

    elif isinstance(img, Image.Image):
        img = np.array(img)

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    return img


def make_train_transform_dino(img_size, complex_augs):
    if complex_augs:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),

            v2.Lambda(apply_clahe_cv2_dino),

            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),
            v2.ColorJitter(brightness=0.1, contrast=0.1),

            v2.ToDtype(torch.float32, scale=True),

            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
    else:
        return v2.Compose([v2.ToImage(), v2.Resize((img_size, img_size), antialias=True),
                           v2.ToDtype(torch.float32, scale=True),
                           v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
                           ])


def make_val_transform_dino(img_size, complex_augs):
    if complex_augs:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),

            v2.Lambda(apply_clahe_cv2_dino),

            v2.ToDtype(torch.float32, scale=True),

            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
        ])