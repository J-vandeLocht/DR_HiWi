import torch
import pandas as pd
import os
from tqdm import tqdm
import argparse
from pathlib import Path

from data.dataset import MILVideoDataset
from dino.utils import make_val_transform_dino
from dino.train_dino_mil import DinoMIL
from dino.train_dino_transformer import DinoSelfAttention

from dino.train_dino_classifier import DinoClassifier


def run_inference_mil(model, loader, device):
    """MIL inference: Evaluates the entire bag of frames at once."""
    model.eval()
    results = {}
    dataset_samples = loader.dataset.data

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="MIL Inference")):
            inputs = batch[0].squeeze(0).to(device)
            labels = batch[1]
            grade = batch[2]
            video_name = dataset_samples[i]['vid_name']

            logits, _ = model(inputs)
            prob = torch.sigmoid(logits).item()

            results[video_name] = {
                "label": int(labels.item()),
                "prob": prob,
                "grade": int(grade.item())
            }
    return results


def run_inference_transformer(model, loader, device):
    """Transformer inference: Evaluates the entire bag of frames at once."""
    model.eval()
    results = {}
    dataset_samples = loader.dataset.data

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Transformer Inference")):
            inputs = batch[0].squeeze(0).to(device)
            labels = batch[1]
            grade = batch[2]
            video_name = dataset_samples[i]['vid_name']

            logits, _ = model(inputs)
            prob = torch.sigmoid(logits).item()

            results[video_name] = {
                "label": int(labels.item()),
                "prob": prob,
                "grade": int(grade.item())
            }
    return results


def run_inference_transformer_rope(model, loader, device):
    model.eval()
    results = {}
    dataset_samples = loader.dataset.data

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Transformer-Rope Inference")):
            inputs = batch[0].squeeze(0).to(device)
            labels = batch[1]
            grade = batch[2]
            frame_positions = batch[3]
            video_name = dataset_samples[i]['vid_name']

            logits, _ = model(inputs, frame_positions)
            prob = torch.sigmoid(logits).item()

            results[video_name] = {
                "label": int(labels.item()),
                "prob": prob,
                "grade": int(grade.item())
            }
    return results


def run_inference_classifier(model, loader, device):
    """Classifier inference: Predicts on each frame, then averages the probabilities."""
    model.eval()
    results = {}
    dataset_samples = loader.dataset.data

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Classifier Inference")):
            # inputs shape: (32, C, H, W)
            inputs = batch[0].squeeze(0).to(device)
            labels = batch[1]
            grade = batch[2]
            video_name = dataset_samples[i]['vid_name']

            # Forward pass: 32 individual frames at once
            logits = model(inputs)  # Shape: (32, 1)
            probs = torch.sigmoid(logits)

            # Average the probabilities across all 32 frames
            avg_prob = probs.mean().item()

            results[video_name] = {
                "label": int(labels.item()),
                "prob": avg_prob,
                "grade": int(grade.item())
            }
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    for split in [1, 2, 3, 4, 5]:
        csv_path = os.path.join(args.output_dir, f"predictions_split_{split}_{args.dataset_name}.csv")

        val_ds = MILVideoDataset(
            json_path=os.path.join(args.annotations_path, f"split_{split}", f"mil_val_{args.dataset_name}.json"),
            num_frames=32,
            transform=val_trans,
            search_dir_paths=[args.videos_path],
        )

        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        run_name = f"split_{split}"
        print(f"\n>>> Evaluating Run: {run_name}")

        # --- 1. Evaluate Transformer Model ---
        trans_model = DinoSelfAttention(checkpoint_path=args.trans_paths[split - 1]).to(device)
        trans_results = run_inference_transformer(trans_model, val_loader, device)

        # --- 2. Evaluate MIL Model ---
        mil_model = DinoMIL(checkpoint_path=args.mil_paths[split - 1]).to(device)
        mil_results = run_inference_mil(mil_model, val_loader, device)

        # --- 3. Evaluate Classifier Model ---
        # Initialize with num_classes=1 so the architecture exactly matches the saved binary head
        classifier = DinoClassifier(freeze_backbone=True, num_classes=1).to(device)
        state_dict = torch.load(args.classifier_paths[split - 1], map_location=device)
        classifier.load_state_dict(state_dict)

        clf_results = run_inference_classifier(classifier, val_loader, device)

        # --- 4. Merge Results into a Single DataFrame ---
        df_mil = pd.DataFrame.from_dict(mil_results, orient='index')
        df_mil = df_mil.rename(columns={'prob': f"MIL_{run_name}"})

        df_clf = pd.DataFrame.from_dict(clf_results, orient='index')
        df_clf = df_clf.rename(columns={'prob': f"Clf_{run_name}"})

        df_trans = pd.DataFrame.from_dict(trans_results, orient='index')
        df_trans = df_trans.rename(columns={'prob': f"Trans_{run_name}"})

        # Drop the redundant 'label' and 'grade' columns from the classifier and transformer dataframe before joining
        df_clf = df_clf.drop(columns=['label']).drop(columns=['grade'])
        df_trans = df_trans.drop(columns=['label']).drop(columns=['grade'])

        # Join both frames on the video name index
        master_df = df_mil.join(df_clf).join(df_trans)
        master_df.index.name = 'video'

        # Reorder columns to look clean: [label, MIL_split_X, Clf_split_X, Trans_split_X]
        master_df = master_df[['label', f"MIL_{run_name}", f"Clf_{run_name}", f"Trans_{run_name}", "grade"]]
        master_df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths for MIL models
    parser.add_argument('--mil_paths', nargs='+', required=True, help="List of 5 MIL checkpoint paths")
    # Paths for Transformer models
    parser.add_argument('--trans_paths', nargs='+', required=True, help="List of 5 Transformer checkpoint paths")
    # Paths for Classifier models
    parser.add_argument('--classifier_paths', nargs='+', required=True, help="List of 5 Classifier checkpoint paths")

    parser.add_argument('--dataset_name', type=str, required=True, default="paxos2020")
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2020/cleaned_videos")
    parser.add_argument('--output_dir', type=str, default="multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()
    main(args)
