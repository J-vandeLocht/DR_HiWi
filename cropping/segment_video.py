import cv2
import argparse
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# --- Configuration ---
MODEL_PATH = 'best_model_new.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 320
TARGET_FPS = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, required=True)
    return p.parse_args()


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
    """Returns the cleaned mask (only the largest blob) and its stats."""
    # Ensure binary_mask is formatted properly for OpenCV
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_uint8, connectivity=8)

    if num_labels <= 1:
        # Only background found
        return None, None, None, None, None

    # stats[0] is the background. We want the largest actual feature.
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a new mask containing ONLY the largest component
    clean_mask = np.uint8(labels == largest_label) * 255

    # Extract dimensions and center
    x, y, w, h, area = stats[largest_label]
    cx, cy = centroids[largest_label]

    return clean_mask, w, h, cx, cy


def crop_and_pad(frame, cx, cy, size):
    """Crops a square of `size` around (cx, cy), padding with black if out of bounds."""
    h, w = frame.shape[:2]
    half_size = size // 2

    x1, y1 = int(cx - half_size), int(cy - half_size)
    x2, y2 = x1 + size, y1 + size

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    src_x1, src_x2 = max(0, x1), min(w, x2)
    src_y1, src_y2 = max(0, y1), min(h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    return canvas


def process_video(video_path):
    output_path = f'video_cropped_masked_{video_path.split("/")[-1].split(".")[0]}.mp4'
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip_frames = max(1, int(orig_fps / TARGET_FPS))

    transforms = get_inference_transforms()

    # --- PASS 1: Analysis ---
    print("Pass 1: Analyzing frames, filtering masks, and finding global crop size...")
    frame_data = {}
    global_max_side = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % skip_frames == 0:
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

            # Extract the largest component mask and its properties
            clean_mask, w, h, cx, cy = get_largest_component_mask(binary_mask)

            if clean_mask is not None:
                max_dim = max(w, h)
                global_max_side = max(global_max_side, max_dim)

                # Store the cleaned mask in memory for Pass 2 to save computation time
                frame_data[frame_idx] = {
                    'cx': cx,
                    'cy': cy,
                    'mask': clean_mask
                }

        frame_idx += 1

    if global_max_side == 0:
        print("Model found absolutely nothing in the entire video.")
        return

    print(f"Calculated Global Crop Size: {global_max_side}x{global_max_side}")

    # --- PASS 2: Cropping & Export ---
    print("Pass 2: Applying clean masks and exporting cropped video...")
    final_crop_size = int(global_max_side * 1.02)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (final_crop_size, final_crop_size))

    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # If the frame was successfully tracked in Pass 1
        if frame_idx in frame_data:
            data = frame_data[frame_idx]
            cx, cy, clean_mask = data['cx'], data['cy'], data['mask']

            # Black out everything except the isolated retina component
            masked_frame = cv2.bitwise_and(frame, frame, mask=clean_mask)

            # Crop the masked frame using the global size
            cropped_frame = crop_and_pad(masked_frame, cx, cy, final_crop_size)
            out.write(cropped_frame)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished! Clean, cropped video saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    process_video(path)