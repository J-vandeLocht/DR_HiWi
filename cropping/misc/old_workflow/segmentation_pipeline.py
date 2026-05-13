import os
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# --- DATASET SELECTION ---
# Set this to "NEW" for your video-extracted frames, or "OLD" for your original set
DATASET_MODE = "NEW"

if DATASET_MODE == "NEW":
    IMG_DIR = Path('misc/frames_to_annotate')
    CSV_PATH = Path('annotations_video_frames.csv')
else:
    IMG_DIR = Path('../data/2024_Paxos_Frames/frames')
    CSV_PATH = Path('annotations.csv')

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
IMG_SIZE = 320  # U-Net++ requires multiples of 32
MODEL_PATH = f'best_model_{DATASET_MODE.lower()}.pth'


# --- 1. The Dataset Class ---
class FundusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['filename']

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not find {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate Mask from CSV coordinates
        # Since images are square, (cx, cy) and (rx, ry) are relative to original square size
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        center = (int(row['center_x']), int(row['center_y']))
        axes = (int(row['radius_x']), int(row['radius_y']))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # [1, 320, 320]

        return image, mask


# --- 2. Transformations ---
def get_transforms(train=True):
    list_trans = [albu.Resize(IMG_SIZE, IMG_SIZE)]

    if train:
        list_trans.extend([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.2),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ])

    list_trans.extend([
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return albu.Compose(list_trans)


# --- 3. Training Loop ---
def run_training():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Check your DATASET_MODE.")
        return

    df = pd.read_csv(CSV_PATH)
    # Split
    train_df = df.sample(frac=0.85, random_state=42)
    valid_df = df.drop(train_df.index)

    print(f"Mode: {DATASET_MODE} | Total: {len(df)} | Train: {len(train_df)} | Val: {len(valid_df)}")

    train_ds = FundusDataset(train_df, IMG_DIR, get_transforms(train=True))
    valid_ds = FundusDataset(valid_df, IMG_DIR, get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize Model
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    best_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = []

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_fn(output, masks)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # Validation
        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                output = model(imgs)
                pred = (torch.sigmoid(output) > 0.5).long()

                s = smp.metrics.get_stats(pred, masks.long(), mode='binary')
                tp += s[0].sum();
                fp += s[1].sum();
                fn += s[2].sum();
                tn += s[3].sum()

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {np.mean(train_loss):.4f} | Val IoU: {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Saved Best Model for {DATASET_MODE}")


if __name__ == "__main__":
    run_training()