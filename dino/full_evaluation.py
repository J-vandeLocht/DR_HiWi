import os
import cv2
import json
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    precision_recall_curve, auc
from .utils import make_val_transform_dino
from .train_dino_mil import DinoMIL


@torch.no_grad()
def get_all_video_features(vid_path, model, transform, device, chunk_size=16):
    """ Reads all frames, applies transform, and extracts DINO features in chunks to prevent OOM. """
    cap = cv2.VideoCapture(vid_path)
    frames = []
    h_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        if transform:
            img = transform(img)

        frames.append(img)

        # Process in chunks
        if len(frames) == chunk_size:
            batch = torch.stack(frames).to(device)
            h = model.extract_features(batch)
            h_list.append(h.cpu())  # Store on CPU RAM temporarily
            frames = []

    # Process any remaining frames
    if len(frames) > 0:
        batch = torch.stack(frames).to(device)
        h = model.extract_features(batch)
        h_list.append(h.cpu())

    cap.release()

    if len(h_list) == 0:
        return None

    # Return all features as a single tensor on the GPU
    return torch.cat(h_list, dim=0).to(device)


def calculate_metrics(y_true, y_probs, name):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs > 0.5).astype(float)

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = auc(rec_curve, prec_curve)

    print(f"[{name}] PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")
    return pr_auc


def run_4way_evaluation(json_path, model_path, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init Model
    model = DinoMIL(checkpoint_path=model_path).to(device)
    model.eval()

    # Parse JSON
    with open(json_path, 'r') as f:
        label_map = json.load(f)

    search_dirs = [
        Path('data/own_clips_hd/train_videos/cleaned_videos'),
        Path('data/own_clips_hd/val_videos/cleaned_videos')
    ]
    all_video_paths = []
    for d in search_dirs:
        if d.exists():
            all_video_paths.extend(list(d.glob('*.mp4')))

    # Results Tracking
    y_true = []
    results = {'A_Max': [], 'B_Uniform': [], 'C_All': [], 'D_Ensemble': []}

    print("Starting 4-Way Evaluation...")

    with torch.no_grad():
        for vid_name, grade in label_map.items():
            base_name = vid_name.replace('.mp4', '').replace('.avi', '')
            pattern = re.compile(rf"{re.escape(base_name)}(?![a-zA-Z0-9])")

            matched_path = next((p for p in all_video_paths if pattern.search(p.name)), None)

            if not matched_path or matched_path.stat().st_size <= 1000:
                continue

            label = 1.0 if grade >= 2 else 0.0

            # 1. Get DINO features for EVERY frame
            h_all = get_all_video_features(str(matched_path), model, transform, device)
            if h_all is None: continue

            N = h_all.shape[0]  # Total frames in video
            y_true.append(label)

            # --- STRATEGY C: All Frames ---
            logits_C, _, _ = model.forward_head(h_all)
            results['C_All'].append(torch.sigmoid(logits_C).item())

            # --- STRATEGY A: Max Attention ---
            # Get raw scores for all frames, pick the highest, run head on just that one feature
            _, _, raw_a = model.forward_head(h_all)
            max_idx = torch.argmax(raw_a)
            logits_A, _, _ = model.forward_head(h_all[max_idx:max_idx + 1])
            results['A_Max'].append(torch.sigmoid(logits_A).item())

            # --- STRATEGY B: Uniform 32 (Current Baseline) ---
            indices_B = np.linspace(0, N - 1, 32, dtype=int)
            logits_B, _, _ = model.forward_head(h_all[indices_B])
            results['B_Uniform'].append(torch.sigmoid(logits_B).item())

            # --- STRATEGY D: Monte Carlo Ensemble (10x Random 32) ---
            d_probs = []
            for _ in range(10):
                # replace=True ensures it works even if video has fewer than 32 frames
                indices_D = np.random.choice(N, 32, replace=True)
                logits_D, _, _ = model.forward_head(h_all[indices_D])
                d_probs.append(torch.sigmoid(logits_D).item())

            results['D_Ensemble'].append(np.mean(d_probs))

            print(f"Processed {base_name} | Frames: {N}")

    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    print("=" * 40)

    calculate_metrics(y_true, results['A_Max'], "Strategy A (Max Attn Frame)")
    calculate_metrics(y_true, results['B_Uniform'], "Strategy B (Uniform 32)  ")
    calculate_metrics(y_true, results['C_All'], "Strategy C (All Frames)    ")
    calculate_metrics(y_true, results['D_Ensemble'], "Strategy D (10x Ensemble)  ")


if __name__ == "__main__":
    val_transform = make_val_transform_dino(512, True)

    json_path = "data/stratified_splits/split_1/mil_val.json"
    model_path = "mil/models/dino_mil_complex_split_1/best_mil_model.pth"

    run_4way_evaluation(json_path, model_path, val_transform)