import cv2
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from misc.utils import apply_clahe_cv2
from informative_frames.crop_frames import crop_and_pad, get_largest_component_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Full Ensemble Production Pipeline")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None, help="Max videos per worker.")
    return parser.parse_args()


def get_seg_transforms(size=512):
    return albu.Compose([
        albu.Resize(size, size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def main():
    args = parse_args()
    DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    VIDEO_DIR = Path('temp/data')
    SEG_MODEL_DIR = Path('cropping/models')
    CLS_MODEL_DIR = Path('informative_frames/models')

    OUT_BASE = Path('temp/data/ensemble_results')
    TXT_DIR = OUT_BASE / 'txt_files'
    PLOT_DIR = OUT_BASE / 'plots'
    VID_DIR = OUT_BASE / 'cleaned_videos'

    for d in [TXT_DIR, PLOT_DIR, VID_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    IMG_SIZE = 512
    CONF_THRESHOLD = 0.5

    # 1. Load Ensembles
    print(f"Loading 5-Fold Unet++ (B4) and Classification (B4) on {DEVICE}...")
    seg_ensemble = []
    cls_ensemble = []
    for fold in range(5):
        # Seg Model
        sm = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1).to(DEVICE).eval()
        sm.load_state_dict(torch.load(list(SEG_MODEL_DIR.glob(f"*fold_{fold}*"))[0], map_location=DEVICE))
        seg_ensemble.append(sm)
        # Cls Model
        cm = models.efficientnet_b4()
        cm.classifier[1] = torch.nn.Linear(cm.classifier[1].in_features, 1)
        cm.load_state_dict(torch.load(list(CLS_MODEL_DIR.glob(f"*Fold{fold}*"))[0], map_location=DEVICE))
        cm.to(DEVICE).eval()
        cls_ensemble.append(cm)

    seg_trans = get_seg_transforms(IMG_SIZE)
    cls_trans = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- File Discovery ---
    all_vids = sorted(list(VIDEO_DIR.glob("*.MOV")))
    my_vids = [v for i, v in enumerate(all_vids) if i % args.total_gpus == args.gpu_id]
    if args.limit: my_vids = my_vids[:args.limit]

    print(f"Worker {args.gpu_id}: Processing {len(my_vids)} videos.")

    for v_path in tqdm(my_vids, desc=f"Overall GPU {args.gpu_id}"):
        vid_id = v_path.stem
        out_vid_path = VID_DIR / f"CLEAN_{vid_id}.mp4"

        if out_vid_path.exists(): continue

        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # PASS 1: Ensemble Analysis
        frame_results = []
        global_max_dim = 0

        with tqdm(total=total_f, desc=f"Analysis: {vid_id}", leave=False) as pbar:
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break

                h, w = frame.shape[:2]
                seg_in = seg_trans(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))['image'].unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # Average Mask
                    m_preds = torch.stack([torch.sigmoid(m(seg_in)) for m in seg_ensemble])
                    mean_mask = torch.mean(m_preds, dim=0).squeeze().cpu().numpy()

                full_mask = cv2.resize(mean_mask, (w, h))
                clean_mask, mw, mh, cx, cy = get_largest_component_mask(full_mask > 0.5)

                prob = 0.0
                if clean_mask is not None:
                    global_max_dim = max(global_max_dim, max(mw, mh))
                    p = int(max(mw, mh) * 0.05)
                    x1, y1 = int(max(0, cx - mw // 2 - p)), int(max(0, cy - mh // 2 - p))
                    crop = frame[y1:y1 + mh + 2 * p, x1:x1 + mw + 2 * p]

                    if crop.size > 0:
                        cls_in = cls_trans(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(
                            DEVICE)
                        with torch.no_grad():
                            # Average Probability
                            c_preds = torch.stack([torch.sigmoid(m(cls_in)) for m in cls_ensemble])
                            prob = torch.mean(c_preds).item()

                frame_results.append({'f': f_idx, 'prob': prob, 'cx': cx, 'cy': cy, 'mask': clean_mask})
                f_idx += 1
                pbar.update(1)

        # PASS 2: Export Video
        valid_f_count = sum(1 for r in frame_results if r['prob'] >= CONF_THRESHOLD and r['mask'] is not None)
        if global_max_dim > 0 and valid_f_count > 0:
            final_sz = int(global_max_dim * 1.05)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (final_sz, final_sz))

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for res in frame_results:
                ret, frame = cap.read()
                if not ret: break
                if res['prob'] >= CONF_THRESHOLD and res['mask'] is not None:
                    masked = cv2.bitwise_and(frame, frame, mask=res['mask'])
                    out.write(crop_and_pad(masked, res['cx'], res['cy'], final_sz))
            out.release()

        # PASS 3: Txt & Plot
        df_res = pd.DataFrame([{'frame': r['f'], 'prob': r['prob']} for r in frame_results])
        df_res.to_csv(TXT_DIR / f"{vid_id}.txt", sep='\t', index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(df_res['frame'], df_res['prob'], label='Ensemble Prob')
        plt.axhline(y=CONF_THRESHOLD, color='r', linestyle='--', label='Threshold')
        plt.title(f"Informative Frames: {vid_id}")
        plt.xlabel("Frame Index")
        plt.ylabel("Avg Probability")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(PLOT_DIR / f"{vid_id}.png")
        plt.close()

        cap.release()


if __name__ == "__main__":
    main()
