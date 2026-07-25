import cv2
import torch
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2

# Assuming get_largest_component_mask is located in your local cropping module/utils
from informative_frames.crop_frames import get_largest_component_mask

# IMG_DIR = Path('data/2024_Paxos_Frames/frames')
# CROPPED_DIR = Path('data/2024_Paxos_Frames/cropped_frames')
IMG_DIR = Path('classifier/frames_sampled')
CROPPED_DIR = Path('classifier/cropped_classifier_frames')
SEG_MODEL_DIR = Path('cropping/models')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 512

CROPPED_DIR.mkdir(parents=True, exist_ok=True)


def get_transforms():
    return albu.Compose([
        albu.Resize(IMG_SIZE, IMG_SIZE),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def load_segmentation_ensemble():
    # Loads the 5-Fold Unet++ ensemble.
    print(f"Loading 5-Fold Unet++ (B4) ensemble on {DEVICE}...")
    seg_ensemble = []
    for fold in range(5):
        sm = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1).to(DEVICE).eval()

        model_paths = list(SEG_MODEL_DIR.glob(f"*fold_{fold}*"))
        if not model_paths:
            raise FileNotFoundError(f"Could not find model for fold {fold} in {SEG_MODEL_DIR}")

        sm.load_state_dict(torch.load(model_paths[0], map_location=DEVICE))
        seg_ensemble.append(sm)
    return seg_ensemble


def save_cropped_images(seg_ensemble):
    # Performs ensemble inference and saves only the masked (cropped) retina image at its original resolution.
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

            # Clean noise & find bounding info on largest component
            clean_mask, mw, mh, cx, cy = get_largest_component_mask(full_mask > 0.5)

            if clean_mask is None:
                print(f"Warning: No valid retina component found for {img_path.name}. Skipping...")
                continue

            # Apply mask to original image to remove non-retina background
            binary_mask = (clean_mask > 0).astype(np.uint8)
            masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=binary_mask)

            # Calculate 5% padding and crop boundaries centered on (cx, cy)
            p = int(max(mw, mh) * 0.05)
            x1 = int(max(0, cx - mw // 2 - p))
            y1 = int(max(0, cy - mh // 2 - p))
            x2 = int(min(w, x1 + mw + 2 * p))
            y2 = int(min(h, y1 + mh + 2 * p))

            cropped_img = masked_img[y1:y2, x1:x2]

            # Save to the new directory
            save_path = CROPPED_DIR / img_path.name
            cv2.imwrite(str(save_path), cropped_img)

            if (i + 1) % 50 == 0:
                print(f"Cropped {i + 1}/{len(all_files)} images...")

    print(f"\nSuccessfully saved cropped images to: {CROPPED_DIR}")


if __name__ == "__main__":
    ensemble = load_segmentation_ensemble()
    save_cropped_images(ensemble)