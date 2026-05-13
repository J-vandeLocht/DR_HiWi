import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2

# --- Configuration ---
IMG_DIR = Path('data/2024_Paxos_Frames/frames')
OUT_DIR = Path('prediction_checks')
CROPPED_DIR = Path('data/2024_Paxos_Frames/cropped_frames_new')
SEG_MODEL_DIR = Path('cropping/models')  # Directory containing the 5 fold models

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 512  # Updated to match the ensemble from script 2

# Ensure directories exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_DIR.mkdir(parents=True, exist_ok=True)


def get_transforms():
    return albu.Compose([
        albu.Resize(IMG_SIZE, IMG_SIZE),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def load_segmentation_ensemble():
    """Loads the 5-Fold Unet++ ensemble."""
    print(f"Loading 5-Fold Unet++ (B4) ensemble on {DEVICE}...")
    seg_ensemble = []
    for fold in range(5):
        sm = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1).to(DEVICE).eval()

        # Find the specific fold model
        model_paths = list(SEG_MODEL_DIR.glob(f"*fold_{fold}*"))
        if not model_paths:
            raise FileNotFoundError(f"Could not find model for fold {fold} in {SEG_MODEL_DIR}")

        sm.load_state_dict(torch.load(model_paths[0], map_location=DEVICE))
        seg_ensemble.append(sm)
    return seg_ensemble


def save_cropped_images(seg_ensemble):
    """
    Performs ensemble inference and saves ONLY the masked (cropped) retina image
    at its original resolution.
    """
    all_files = sorted([f for f in IMG_DIR.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    print(f"Generating cropped frames for {len(all_files)} images...")

    transforms = get_transforms()

    with torch.no_grad():
        for i, img_path in enumerate(all_files):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue

            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Inference
            input_tensor = transforms(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)

            # Average Mask across the ensemble
            m_preds = torch.stack([torch.sigmoid(m(input_tensor)) for m in seg_ensemble])
            mean_mask = torch.mean(m_preds, dim=0).squeeze().cpu().numpy()

            # Resize mask back to ORIGINAL resolution
            full_mask = cv2.resize(mean_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (full_mask > 0.5).astype(np.uint8)

            # Apply mask to original image
            cropped_img = cv2.bitwise_and(img_bgr, img_bgr, mask=binary_mask)

            # Save to the new directory
            save_path = CROPPED_DIR / img_path.name
            cv2.imwrite(str(save_path), cropped_img)

            if (i + 1) % 50 == 0:
                print(f"Cropped {i + 1}/{len(all_files)} images...")

    print(f"\nSuccessfully saved cropped images to: {CROPPED_DIR}")


def run_bulk_check(seg_ensemble):
    """
    Performs ensemble inference and saves a side-by-side comparison
    of the original image and the masked image.
    """
    all_files = sorted([f for f in IMG_DIR.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    print(f"Found {len(all_files)} images. Starting batch prediction check...")

    transforms = get_transforms()

    with torch.no_grad():
        for i, img_path in enumerate(all_files):
            # Load
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue

            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Preprocess & Infer
            input_tensor = transforms(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)

            # Average Mask across the ensemble
            m_preds = torch.stack([torch.sigmoid(m(input_tensor)) for m in seg_ensemble])
            mean_mask = torch.mean(m_preds, dim=0).squeeze().cpu().numpy()

            # Post-process Mask
            full_mask = cv2.resize(mean_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (full_mask > 0.5).astype(np.uint8)

            # Create the "Cropped" version (Bitwise Mask)
            masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=binary_mask)

            # Create Side-by-Side (SBS)
            sbs_view = np.hstack((img_bgr, masked_img))

            # Save
            out_path = OUT_DIR / f"check_{img_path.name}"
            cv2.imwrite(str(out_path), sbs_view)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(all_files)} images...")

    print(f"\nFinished! All check images are in: {OUT_DIR}")


if __name__ == "__main__":
    # Load ensemble once to save memory and time
    ensemble = load_segmentation_ensemble()

    save_cropped_images(ensemble)
    run_bulk_check(ensemble)