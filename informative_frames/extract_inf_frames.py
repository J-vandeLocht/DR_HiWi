import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and crop clean video frames.")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'all'],
                        help="Which dataset split to process.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="The ID of the GPU to use (e.g., 0, 1, 2, 3).")
    parser.add_argument('--total_gpus', type=int, default=1,
                        help="Total number of parallel instances running.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Maximum number of NEW videos to process in this run (per worker).")
    return parser.parse_args()


# --- Preprocessing & Cropping Functions ---
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


def apply_clahe_cv2(img_input):
    # 1. Convert PIL Image (from Resize) to Numpy array for OpenCV
    img_np = np.array(img_input)

    # 2. Apply CLAHE logic
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    processed_np = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    # 3. Convert back to PIL Image so the next transform (ToTensor) works correctly
    return Image.fromarray(processed_np)


def main():
    args = parse_args()

    # --- Configuration ---
    VIDEO_DIR = Path('data/dr_videos')
    SPLIT_CSV = 'informative_frames/split_info_latest.csv'
    SEG_MODEL_PATH = 'cropping/best_model_new.pth'
    CLS_MODEL_PATH = 'informative_frames/best_Informative_B3_20260331_174528.pth'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE_SEG = 320
    CONF_THRESHOLD = 0.7

    OUT_BASE = Path(f'{args.split}_results_new')
    TXT_DIR = OUT_BASE / 'txt_files'
    PLOT_DIR = OUT_BASE / 'plots'
    VID_DIR = OUT_BASE / 'cleaned_videos'

    for d in [TXT_DIR, PLOT_DIR, VID_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"--- Initializing Worker on {DEVICE} ---")

    # Load Models
    seg_model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3, classes=1).to(DEVICE).eval()
    seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))

    efficientnets = {
        'b0': models.efficientnet_b0,
        'b1': models.efficientnet_b1,
        'b2': models.efficientnet_b2,
        'b3': models.efficientnet_b3,
        'b4': models.efficientnet_b4,
        'b5': models.efficientnet_b5,
        'b6': models.efficientnet_b6,
        'b7': models.efficientnet_b7
    }

    model_fn = efficientnets['b3']
    cls_model = model_fn()
    cls_model.classifier[1] = torch.nn.Linear(cls_model.classifier[1].in_features, 1)
    cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    cls_model.to(DEVICE).eval()

    seg_transforms = get_inference_transforms()
    cls_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(SPLIT_CSV)
    if args.split == 'all':
        target_videos = sorted(df['video_id'].unique())
    else:
        target_videos = sorted(df[df['split'] == args.split]['video_id'].unique())

    # Modulo Logic for parallelization
    my_videos = [vid for i, vid in enumerate(target_videos) if i % args.total_gpus == args.gpu_id]

    processed_count = 0
    print(f"Worker {args.gpu_id}: Assigned {len(my_videos)} total. Checking for progress...")

    for vid_id in my_videos:
        # Check if limit is reached
        if args.limit is not None and processed_count >= args.limit:
            print(f"[GPU {args.gpu_id}] Reached specified limit of {args.limit} videos. Stopping.")
            break

        video_path = VIDEO_DIR / f"{vid_id}.mp4"
        output_vid_path = VID_DIR / f"CLEAN_{vid_id}.mp4"

        # RESUME LOGIC: Skip if file already exists
        if output_vid_path.exists():
            print(f"[GPU {args.gpu_id}] Skipping {vid_id} (Already exists).")
            continue

        if not video_path.exists(): continue

        print(f"\n[GPU {args.gpu_id}] Starting: {vid_id} (Video {processed_count + 1})")

        cap = cv2.VideoCapture(str(video_path))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- PASS 1: Analysis ---
        global_max_side = 0
        frame_data = {}
        f_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            sq_frame, y_off, x_off, side = pad_to_square_info(frame)
            input_img = cv2.resize(sq_frame, (IMG_SIZE_SEG, IMG_SIZE_SEG))
            img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_tensor = seg_transforms(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                prob_mask = torch.sigmoid(seg_model(input_tensor)).squeeze().cpu().numpy()

            sq_mask = cv2.resize(prob_mask, (side, side))
            rect_mask = sq_mask[y_off:y_off + height, x_off:x_off + width]
            binary_mask = (rect_mask > 0.5).astype(np.uint8)
            clean_mask, w, h, cx, cy = get_largest_component_mask(binary_mask)

            prob = 0.0
            if clean_mask is not None:
                global_max_side = max(global_max_side, max(w, h))
                masked = cv2.bitwise_and(frame, frame, mask=clean_mask)
                dim = int(max(w, h) * 1.05)
                x1, y1 = int(max(0, cx - dim // 2)), int(max(0, cy - dim // 2))
                cropped_cls = masked[y1:y1 + dim, x1:x1 + dim]

                if cropped_cls.size > 0:
                    cropped_cls = cv2.cvtColor(cropped_cls, cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                        prob = torch.sigmoid(
                            cls_model(cls_transforms(Image.fromarray(cropped_cls)).unsqueeze(0).to(DEVICE))).item()

            frame_data[f_idx] = {'prob': prob, 'cx': cx, 'cy': cy, 'mask': clean_mask}
            f_idx += 1

        if global_max_side == 0:
            cap.release()
            continue

        # --- PASS 2: Export ---
        final_crop_size = int(global_max_side * 1.02)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(output_vid_path), fourcc, orig_fps, (final_crop_size, final_crop_size))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            data = frame_data.get(f_idx)
            if data and data['prob'] >= CONF_THRESHOLD and data['mask'] is not None:
                masked_frame = cv2.bitwise_and(frame, frame, mask=data['mask'])
                cropped_frame = crop_and_pad(masked_frame, data['cx'], data['cy'], final_crop_size)
                out_video.write(cropped_frame)
            f_idx += 1

        cap.release()
        out_video.release()

        # Save Txt and Plot
        df_res = pd.DataFrame([{'frame': k, 'prob': v['prob']} for k, v in frame_data.items()])
        df_res.to_csv(TXT_DIR / f"{vid_id}.txt", sep='\t', index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(df_res['frame'], df_res['prob'])
        plt.axhline(y=CONF_THRESHOLD, color='r', linestyle='--')
        plt.savefig(PLOT_DIR / f"{vid_id}.png")
        plt.close()

        processed_count += 1

    print(f"\nWorker {args.gpu_id} done.")


if __name__ == "__main__":
    main()
