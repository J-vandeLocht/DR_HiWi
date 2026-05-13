import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import shutil

# --- Configuration ---
LABEL_CSV = 'manual_labels_binary_2.csv'  # The CSV from your manual labeling step
INPUT_DIR = Path('data/frames_raw_extract_2')  # Where your raw PNG/JPG frames are
OUTPUT_DIR = Path('data/labeled_and_cropped_dataset_2')
MODEL_PATH = 'cropping/best_model_new.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 320  # Model input size

# --- Setup Output Folders ---
(OUTPUT_DIR / '1').mkdir(parents=True, exist_ok=True)  # Informative
(OUTPUT_DIR / '0').mkdir(parents=True, exist_ok=True)  # Junk


def get_inference_transforms():
    return albu.Compose([
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def pad_to_square_info(image):
    h, w = image.shape[:2]
    side = max(h, w)
    pad_img = np.zeros((side, side, 3), dtype=np.uint8)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    pad_img[y_off:y_off + h, x_off:x_off + w] = image
    return pad_img, y_off, x_off, side


def get_largest_component_mask(binary_mask):
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_uint8, connectivity=8)
    if num_labels <= 1: return None, None, None, None, None

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = np.uint8(labels == largest_label) * 255
    x, y, w, h, area = stats[largest_label]
    cx, cy = centroids[largest_label]
    return clean_mask, w, h, cx, cy


def crop_and_pad(frame, cx, cy, size):
    h, w = frame.shape[:2]
    half_size = int(size // 2)
    x1, y1 = int(cx - half_size), int(cy - half_size)
    x2, y2 = x1 + size, y1 + size
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    src_x1, src_x2 = max(0, x1), min(w, x2)
    src_y1, src_y2 = max(0, y1), min(h, y2)
    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
    return canvas


def process_labeled_frames():
    # Load Model
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    transforms = get_inference_transforms()
    df = pd.read_csv(LABEL_CSV)

    print(f"Starting cropping for {len(df)} labeled images...")

    for _, row in df.iterrows():
        img_name = row['filename']
        print(img_name)
        label = str(int(row['label']))
        img_path = INPUT_DIR / img_name

        if not img_path.exists():
            print(f"Warning: {img_name} not found in input directory.")
            continue

        frame = cv2.imread(str(img_path))
        height, width = frame.shape[:2]

        # 1. Masking Logic (Pass 1 style)
        sq_frame, y_off, x_off, side = pad_to_square_info(frame)
        input_img = cv2.resize(sq_frame, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_tensor = transforms(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            prob_mask = torch.sigmoid(logits).squeeze().cpu().numpy()

        sq_mask = cv2.resize(prob_mask, (side, side))
        rect_mask = sq_mask[y_off:y_off + height, x_off:x_off + width]
        binary_mask = (rect_mask > 0.5).astype(np.uint8)

        clean_mask, w, h, cx, cy = get_largest_component_mask(binary_mask)

        if clean_mask is not None:
            # 2. Mask and Crop Logic (Pass 2 style)
            masked_frame = cv2.bitwise_and(frame, frame, mask=clean_mask)

            # Use the larger dimension of the mask + 5% padding for the square crop
            crop_dim = int(max(w, h) * 1.05)
            cropped_img = crop_and_pad(masked_frame, cx, cy, crop_dim)

            # 3. Save as Lossless PNG
            save_name = img_path.stem + ".png"
            target_path = OUTPUT_DIR / label / save_name
            cv2.imwrite(str(target_path), cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            print(f"Skipping {img_name}: No retina detected by model.")

    print(f"\nSuccess! Labeled and cropped dataset is in: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    process_labeled_frames()